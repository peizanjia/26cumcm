"""Audit the frozen independent validation using complete whole-scene records.

No simulator calls and no selected report examples. Component means are computed
by dividing each scene's component by its own true source count first.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import zipfile

import numpy as np
from scipy.stats import t as student_t


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def statistics(values):
    a = np.asarray(values, dtype=float)
    if not len(a):
        return dict(n=0, mean=None, mean_ci95=None)
    n = len(a)
    average = float(a.mean())
    se = float(a.std(ddof=1) / math.sqrt(n)) if n > 1 else None
    half = float(student_t.ppf(.975, n - 1) * se) if n > 1 else None
    return dict(n=n, mean=average, standard_error=se,
        mean_ci95=[average - half, average + half] if half is not None else None,
        min=float(a.min()), median=float(np.median(a)),
        p90=float(np.quantile(a, .9)), p95=float(np.quantile(a, .95)), max=float(a.max()))


def group_audit(rows):
    complete = [r for r in rows if r['complete']]
    values = [r['average_time_s'] for r in complete]
    strata = defaultdict(list)
    parameter_sets = {}
    for row in rows:
        params = row.get('parameters', {})
        key = hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()
        parameter_sets.setdefault(key, dict(parameters=params, seeds=[]))['seeds'].append(row['seed'])
        strata[row['true_total']].append(row)
    components = {key: statistics([row['time_components_s'][key] / row['true_total']
                                   for row in complete])
                  for key in ('movement', 'measure', 'switch', 'clear')}
    below = [r['seed'] for r in complete if r['average_time_s'] < 200.]
    cost_residual = max((abs(sum(row['time_components_s'].values()) - row['virtual_time_s'])
                         for row in rows), default=0.)
    return dict(count=len(rows), complete=len(complete), failures=len(rows) - len(complete),
        seeds=[r['seed'] for r in rows],
        true_sources=sum(r['true_total'] for r in rows), true_cleared=sum(r['true_cleared'] for r in rows),
        time_s_per_source=statistics(values),
        timing_conditional_on_completion=len(rows) != len(complete),
        below_200_count=len(below), below_200_seeds=below,
        below_200_fraction_all_scenes=len(below) / len(rows) if rows else None,
        zero_failure_one_sided_upper95=1 - .05 ** (1 / len(rows)) if rows and len(rows) == len(complete) else None,
        source_count_strata={str(n): dict(count=len(items), complete=sum(r['complete'] for r in items),
            seeds=[r['seed'] for r in items],
            time_s_per_source=statistics([r['average_time_s'] for r in items if r['complete']]))
                            for n, items in sorted(strata.items())},
        component_s_per_source=components,
        component_mean_sum_s_per_source=sum(row['mean'] for row in components.values()),
        maximum_component_accounting_residual_s=cost_residual,
        mean_distance_m=float(np.mean([r['distance_m'] for r in rows])),
        mean_measures=float(np.mean([r['measures'] for r in rows])),
        long_reversals=sum(r.get('long_reversals', 0) for r in rows),
        mean_wall_s=float(np.mean([r['wall_time_s'] for r in rows])),
        parameter_sets=parameter_sets,
        failed_cases=[dict(seed=r['seed'], failure=r.get('failure')) for r in rows if not r['complete']])


def paired(left_rows, right_rows):
    left, right = ({r['seed']: r for r in rows} for rows in (left_rows, right_rows))
    common = sorted(set(left) & set(right))
    seeds = [s for s in common if left[s]['complete'] and right[s]['complete']]
    differences = [left[s]['average_time_s'] - right[s]['average_time_s'] for s in seeds]
    a = float(np.mean([left[s]['average_time_s'] for s in seeds]))
    b = float(np.mean([right[s]['average_time_s'] for s in seeds]))
    components = {key: statistics([left[s]['time_components_s'][key] / left[s]['true_total']
                                   - right[s]['time_components_s'][key] / right[s]['true_total']
                                   for s in seeds]) for key in ('movement', 'measure', 'switch', 'clear')}
    return dict(sign='left_minus_right; negative means left faster', paired_seeds=seeds,
        excluded_incomplete_seeds=[s for s in common if s not in seeds],
        difference_s_per_source=statistics(differences),
        faster=sum(x < -1e-9 for x in differences), same=sum(abs(x) <= 1e-9 for x in differences),
        slower=sum(x > 1e-9 for x in differences),
        left_mean_s_per_source=a, right_mean_s_per_source=b,
        saving_percent_ratio_of_means=100. * (b - a) / b,
        saving_percent_scene_average=statistics([100. * (right[s]['average_time_s'] - left[s]['average_time_s'])
                                                / right[s]['average_time_s'] for s in seeds]),
        component_differences_s_per_source=components,
        raw_pairs=[dict(seed=s, source_count=left[s]['true_total'], left=left[s]['average_time_s'],
                        right=right[s]['average_time_s'], difference=left[s]['average_time_s'] - right[s]['average_time_s'])
                   for s in seeds])


def audit_snapshot(path, recorded_hash):
    if not path.exists():
        return dict(available=False)
    h = hashlib.sha256()
    entries = []
    with zipfile.ZipFile(path) as archive:
        # The benchmark wrote sorted source files in order and hashed Windows
        # relative path strings, while ZIP entry names use forward slashes.
        for info in archive.infolist():
            if info.is_dir():
                continue
            data = archive.read(info.filename)
            h.update(info.filename.replace('/', '\\').encode())
            h.update(data)
            entries.append(dict(path=info.filename, sha256=hashlib.sha256(data).hexdigest()))
    return dict(available=True, archive_sha256=sha256(path), file_count=len(entries),
                reconstructed_policy_hash=h.hexdigest(), matches_recorded_policy_hash=h.hexdigest() == recorded_hash,
                files=entries)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', default='question3/global_policy/adaptive_mpc/outputs/validation')
    parser.add_argument('--output', help='Default: <input>/final_audit.json')
    args = parser.parse_args()
    root = Path(args.input)
    cases_path, summary_path, manifest_path = (root / name for name in ('cases.json', 'summary.json', 'manifest.json'))
    cases = json.loads(cases_path.read_text(encoding='utf-8'))
    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    groups = {variant: group_audit(rows) for variant, rows in cases.items()}
    checks = {}
    for variant, group in groups.items():
        expected = summary[variant]
        check = dict(count_matches=group['count'] == expected['count'],
            completion_matches=group['complete'] == expected['complete'],
            mean_matches=abs(group['time_s_per_source']['mean'] - expected['mean_s_per_source']) < 1e-8,
            cost_means_add_up=abs(group['component_mean_sum_s_per_source'] - group['time_s_per_source']['mean']) < 1e-6,
            seeds_match_manifest=group['seeds'] == manifest['seeds'])
        if not all(check.values()):
            raise AssertionError((variant, check))
        checks[variant] = check
    comparisons = {}
    for left, right in (('tour_candidate', 'combined'), ('tour_candidate', 'sweep_candidate'),
                        ('sweep_candidate', 'combined'), ('lean', 'combined')):
        comparisons[f'{left}_minus_{right}'] = paired(cases[left], cases[right])
    tour_mean = groups['tour_candidate']['time_s_per_source']['mean']
    result = dict(evaluation='independent_local_synthetic_validation',
        statistical_unit='whole_scene', source_files={str(p): sha256(p) for p in (cases_path, summary_path, manifest_path)},
        recorded_policy_hash=manifest.get('source_hash'), manifest=manifest,
        source_snapshot=audit_snapshot(root / 'source_snapshot.zip', manifest.get('source_hash')),
        interpretation=dict(mean_ci='Student-t 95 percent interval of whole-scene mean; not a future-scene prediction interval',
            paired_ci='Marginal paired Student-t 95 percent; comparisons not multiplicity-adjusted',
            component_means='For each scene divide its component seconds by that scene source count, then average scenes',
            saving_percent='Primary percentage is (right mean - left mean) / right mean; scene-average ratios are separate',
            historical_1000='Root comparison.html is not included in any result below',
            generalization='Local simulator distribution only; no official score or completion guarantee inferred'),
        groups=groups, paired=comparisons, consistency_checks=checks,
        target=dict(threshold_s_per_source=200., achieved=tour_mean < 200.,
                    remaining_s_per_source=tour_mean - 200.,
                    remaining_reduction_percent=100. * (tour_mean - 200.) / tour_mean))
    output = Path(args.output) if args.output else root / 'final_audit.json'
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(dict(output=str(output), snapshot_matches=result['source_snapshot'].get('matches_recorded_policy_hash'),
                         group_means={k: v['time_s_per_source'] for k, v in groups.items()},
                         below_200={k: v['below_200_count'] for k, v in groups.items()}, target=result['target']), indent=2))


if __name__ == '__main__':
    main()

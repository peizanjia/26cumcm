"""Build a standalone, truth-hidden comparison from local paired traces.

Input: cases.json {combined: [summary], adaptive: [summary]}, summary.json,
and <variant>/<seed>.json.gz {summary, commands, events, truth}.
Imported comparison.html is a historical reference, never pooled with new data.
"""
import argparse
import gzip
import json
import math
from pathlib import Path


LABELS = {'combined': '原联合策略', 'adaptive': '全局动作估值',
          'lean': '全局动作估值·精简扫频', 'sweep_candidate': '逐步重排·扇区路线',
          'tour_candidate': '逐步重排·清除与覆盖共路'}


def finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def select_cases(rows, count=2):
    """Deterministic diagnostic selection; never estimate mean from these cases."""
    names = list(rows)
    base = 'combined' if 'combined' in rows else names[0]
    new = 'adaptive' if 'adaptive' in rows else names[-1]
    old = {int(r['seed']): r for r in rows[base]}
    result = {}

    def add(row, reason):
        result.setdefault(int(row['seed']), []).append(reason)

    valid = [r for r in rows[new] if finite(r.get('average_time_s'))]
    for row in rows[new]:
        if not row.get('complete', False):
            add(row, '新策略未完成')
    for row in sorted(valid, key=lambda r: (-r['average_time_s'], r['seed']))[:count]:
        add(row, '新策略最慢')
    paired = [r for r in valid if int(r['seed']) in old
              and finite(old[int(r['seed'])].get('average_time_s'))
              and r.get('complete') and old[int(r['seed'])].get('complete')]
    delta = lambda r: r['average_time_s'] - old[int(r['seed'])]['average_time_s']
    for row in sorted((r for r in paired if delta(r) > 1e-9),
                      key=lambda r: (-delta(r), r['seed']))[:count]:
        add(row, '相对退步最大')
    for row in sorted((r for r in paired if delta(r) < -1e-9),
                      key=lambda r: (delta(r), r['seed']))[:count]:
        add(row, '相对改善最大')
    if valid:
        add(sorted(valid, key=lambda r: (r['average_time_s'], r['seed']))[len(valid) // 2],
            '中位表现参照')
    return result


def parse_reference(path):
    if not path.exists():
        return None
    content = path.read_text(encoding='utf-8')
    if 'const DATA=' not in content:
        return None
    data, _ = json.JSONDecoder().raw_decode(content.split('const DATA=', 1)[1])
    return dict(source=path.name, summary=data.get('summary', {}),
                manifest={k: data.get('manifest', {}).get(k)
                          for k in ('count', 'seed_start', 'evaluation', 'policy_hash')})


def compact_pack(pack):
    commands = []
    for cmd in pack.get('commands', []):
        response = cmd.get('response', {})
        commands.append(dict(path=cmd.get('path'), position=cmd.get('position'),
            channel=cmd.get('channel'), reason=cmd.get('reason', ''),
            time=response.get('virtual_time_s'),
            result=response.get('clear_result', response.get('measure_result', response.get('result'))),
            bearing=response.get('svd_deg', response.get('bearing_deg', response.get('bearing')))))
    phases = {'global_action_decision', 'stop_scan', 'local_decision',
              'route_replan', 'map_replan', 'replan_trigger', 'global_replan',
              'initial_probe_uncommitted', 'linked_terminal', 'global_tour_replan'}
    events = [e for e in pack.get('events', [])
              if e.get('phase') in phases or str(e.get('phase', '')).startswith('stop_scan')]
    return dict(summary=pack['summary'], commands=commands, events=events,
                truth=pack.get('truth', []))


def cost_breakdown(row):
    n = row.get('true_total') or row.get('cleared') or 1
    move = row.get('distance_m', 0.) / 5.
    measure = row.get('measures', 0) * 5.
    clear = row.get('true_cleared', row.get('cleared', 0)) * 5.
    miss = row.get('misses', 0) * 3.
    other = row.get('virtual_time_s', 0) - move - measure - clear - miss
    return {name: value / n for name, value in
            [('移动', move), ('测量', measure), ('成功清除', clear), ('失败清除', miss), ('切频等余额', other)]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', default='question3/global_policy/adaptive_mpc/outputs/validation')
    p.add_argument('--output', help='Default: <input>/report')
    p.add_argument('--reference', default='comparison.html')
    p.add_argument('--case-count', type=int, default=2)
    args = p.parse_args()
    source = Path(args.input)
    out = Path(args.output) if args.output else source / 'report'
    rows = json.loads((source / 'cases.json').read_text(encoding='utf-8'))
    summary = json.loads((source / 'summary.json').read_text(encoding='utf-8'))
    manifest_path = source / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.exists() else {}
    selected = select_cases(rows, max(1, args.case_count))
    scenes = {}
    missing = []
    for seed, reasons in selected.items():
        variants = {}
        for name in rows:
            path = source / name / f'{seed}.json.gz'
            if not path.exists():
                missing.append(str(path))
                continue
            with gzip.open(path, 'rt', encoding='utf-8') as f:
                variants[name] = compact_pack(json.load(f))
            variants[name]['cost_s_per_source'] = cost_breakdown(variants[name]['summary'])
        if variants:
            scenes[str(seed)] = dict(reasons=reasons, variants=variants)
    data = dict(labels={name: LABELS.get(name, name) for name in rows},
                names=list(rows), rows=rows, summary=summary, manifest=manifest,
                scenes=scenes, case_order=list(scenes), missing_traces=missing,
                reference=parse_reference(Path(args.reference)),
                input=str(source), evaluation='local_synthetic_paired')
    out.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(data, ensure_ascii=False, allow_nan=False)
    (out / 'data.json').write_text(serialized, encoding='utf-8')
    template = Path(__file__).with_name('report_template.html').read_text(encoding='utf-8')
    (out / 'comparison.html').write_text(template.replace('__REPORT_DATA__',
        serialized.replace('</', '<\\/')), encoding='utf-8')
    (out / 'case_selection.json').write_text(json.dumps(selected, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(dict(output=str(out / 'comparison.html'), cases=len(scenes),
                         missing_traces=missing), ensure_ascii=False))


if __name__ == '__main__':
    main()

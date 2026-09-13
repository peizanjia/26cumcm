"""Offline full-mission statistics. Truth never enters the running policy."""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import numpy as np

from question4.dynamic_joint.behavior_analysis import reconstruct_behavior


def analyze(path):
    path = Path(path)
    data = json.loads(path.read_text(encoding='utf-8'))
    details, grouped = [], {}
    for row in data['rows']:
        with gzip.open(row['history_file'], 'rt', encoding='utf-8') as f:
            h = json.load(f)
        b = reconstruct_behavior(h['commands'])
        moves = Counter(r['reason'] for r in h['commands'] if r['move_s'] > 1e-7)
        item = dict(seed=row['seed'], strategy=row['strategy'],
                    elastic_updates=h.get('coverage', {}).get('last_plan', {}).get('elastic_updates_total', 0),
                    movement_reasons=dict(moves), side_scan_count=b['side_scan_count'],
                    side_scan_halves=b['side_scan_halves'], side_scan_no_signal=b['side_scan_no_signal'])
        details.append(item)
    for name in data['configs']:
        rows = [r for r in data['rows'] if r['strategy'] == name]
        ds = [d for d in details if d['strategy'] == name]
        values = np.array([r['per_source_s'] for r in rows])
        rng = np.random.default_rng(73119)
        boot = values[rng.integers(0, len(values), (5000, len(values)))].mean(axis=1)
        moves = Counter()
        for d in ds:
            moves.update(d['movement_reasons'])
        grouped[name] = dict(mean_bootstrap95_s=np.quantile(boot, [.025, .975]).tolist(),
            movement_reasons=dict(moves), elastic_updates=sum(d['elastic_updates'] for d in ds),
            mean_elastic_updates=float(np.mean([d['elastic_updates'] for d in ds])),
            side_scan_count=sum(d['side_scan_count'] for d in ds),
            side_scan_halves=sum(d['side_scan_halves'] for d in ds),
            side_scan_no_signal=sum(d['side_scan_no_signal'] for d in ds),
            source_count_groups={str(n): dict(count=len(rs),
                mean_per_source_s=float(np.mean([r['per_source_s'] for r in rs])),
                mean_search_after_last_clear_s=float(np.mean([r['search_after_last_clear_s'] for r in rs])))
                for n in range(10, 17) if (rs := [r for r in rows if r['source_count'] == n])})
    output = dict(scope='offline reconstruction of real public responses',
                  grouped=grouped, details=details)
    (path.parent/'behavior.json').write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding='utf-8')
    return output


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--validation', type=Path, default=Path(__file__).parent/'outputs/validate.json')
    args = p.parse_args()
    result = analyze(args.validation)
    print(json.dumps(result['grouped'], ensure_ascii=False, indent=2))

"""Assemble compact evidence after the frozen independent validation and audit."""
import argparse
import json
import platform
from pathlib import Path

import numpy as np
import scipy


BASE = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def summarize(output):
    output = Path(output)
    final = read(output/'validate.json')
    frozen = read(output/'frozen_parameters.json')
    selection = read(output/'confirmation.json')
    training = read(output/'bayes/checkpoint.json')
    audit = read(output/'audit.json')
    selection_audits = [read(output/path) for path in (
        'confirm_baselines/audit.json', 'confirmation/bayes_candidates_audit.json',
        'confirmation_extra/audit.json')]
    if final['seeds'] != frozen['final_validation_seeds']:
        raise ValueError('Final scenes differ from the frozen registration')
    if final['code_sha256'] != frozen['code_sha256']:
        raise ValueError('Final runtime differs from the frozen runtime')
    if final['configs'] != frozen['configs']:
        raise ValueError('Final configurations differ from the frozen selection')
    expected = {(name, seed) for name in frozen['configs'] for seed in final['seeds']}
    actual = [(row['strategy'], row['seed']) for row in final['rows']]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError('Final paired matrix is incomplete or duplicated')
    if not audit['passed'] or not all(a['passed'] for a in selection_audits):
        raise ValueError('Resolve physical audit failures before publishing')
    mean_intervals = {}
    for name in final['configs']:
        rows = [r for r in final['rows'] if r['strategy'] == name]
        if not all(r['completed'] and r['cleared'] == r['source_count'] for r in rows):
            raise ValueError('Incomplete missions cannot have a complete result')
        values = np.asarray([r['per_source_s'] for r in rows])
        rng = np.random.default_rng(419)
        samples = values[rng.integers(len(values), size=(10000, len(values)))].mean(axis=1)
        mean_intervals[name] = dict(mean_per_source_s=float(values.mean()),
            bootstrap95_s=np.quantile(samples, [.025, .975]).tolist(),
            bootstrap_draws=10000, bootstrap_seed=419)
    result = dict(scope='local synthetic; no official practice or formal test',
        metric='equal-scene mean of complete action time / actual source count; all five costs included',
        software=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                      platform=platform.platform()),
        training=dict(seeds=frozen['training_seeds'], algorithm=training['algorithm'],
            completed_trials=training['completed_trials'], best_trial=training['best_trial'],
            best_mean_per_source_s=training['best_mean_per_source_s'],
            audit=read(output/'bayes_audit.json')),
        structure=read(output/'structure_corrected/develop.json')['summary'],
        selection=dict(seeds=selection['seeds'], configs=len(selection['configs']),
            runs=len(selection['rows']), selected=selection['selected'],
            rule=selection['selection_rule'],
            protocol_amendment='After reading selection bad cases, added half_default and half_bayes027 with future_scan_weight=0.5; selection scenes are adaptive, not final test',
            summary=selection['summary'], paired=selection['paired'],
            audited_commands=sum(a['commands'] for a in selection_audits)),
        final=dict(seeds=final['seeds'], summary=final['summary'], paired=final['paired'],
                   mean_intervals=mean_intervals, audit=audit),
        selected_config=frozen['configs']['tuned_joint'], code_sha256=final['code_sha256'],
        limitations=['finite-range, finite-budget best tested selection; no global optimality proof',
                     'synthetic-distribution estimates, not a guarantee for every source arrangement',
                     'mean intervals and paired improvement intervals answer different questions'])
    path = output/'optimization_summary.json'
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    return dict(file=str(path.resolve()), selected=selection['selected'],
                final=mean_intervals, paired=final['paired'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=BASE/'outputs')
    args = parser.parse_args()
    print(json.dumps(summarize(args.output), ensure_ascii=False, indent=2))

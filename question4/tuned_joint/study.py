"""Promote development candidates and freeze before the untouched test set."""
import argparse
import json
from pathlib import Path

from question4.free_joint.evaluate import aggregate, paired, resolve_configs, write_json, code_hashes

BASE = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def promote(output, count=3):
    state = read(output/'bayes/checkpoint.json')
    if state['status'] != 'complete':
        raise ValueError('Finish the declared Bayesian budget before promotion')
    eligible = sorted((t for t in state['trials'] if t['status'] == 'complete'
                       and t['score']['complete'] and t['proposal']['kind'] != 'baseline'),
                      key=lambda t: t['score']['objective'])[:count]
    configs = {t['name']: dict(t['config'], label=f"贝叶斯候选 {t['name']}") for t in eligible}
    write_json(output/'confirmation_variants.json', dict(configs=configs,
               rule='Top three complete nonbaseline training means; baseline structure and free_joint also confirmed',
               train_seeds=state['settings']['train_seeds'], confirmation_seeds=list(range(20295100, 20295132))))
    return configs


def freeze(output):
    if (output/'validate.json').exists() or (output/'validate_histories').exists():
        raise ValueError('Final validation has started; do not reselect on opened test scenes')
    baseline = read(output/'confirm_baselines/develop.json')
    experiments = [baseline, read(output/'confirmation/develop.json')]
    if (output/'confirmation_extra_variants.json').exists():
        experiments.append(read(output/'confirmation_extra/develop.json'))
    for trial in experiments[1:]:
        if baseline['seeds'] != trial['seeds'] or baseline['code_sha256'] != trial['code_sha256']:
            raise ValueError('Confirmation groups must share scenes and the same frozen runtime')
    if baseline['code_sha256'] != code_hashes():
        raise ValueError('Runtime changed after confirmation')
    configs = {name: spec for trial in experiments for name, spec in trial['configs'].items()}
    rows = [row for trial in experiments for row in trial['rows']]
    expected_seeds = list(range(20295100, 20295132))
    if baseline['seeds'] != expected_seeds:
        raise ValueError('Selection must use the registered 32 scenes')
    if len(configs) != sum(len(trial['configs']) for trial in experiments):
        raise ValueError('Selection configuration names must be unique')
    expected = {(name, seed) for name in configs for seed in expected_seeds}
    actual = [(row['strategy'], row['seed']) for row in rows]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError('Finish the full paired selection matrix before freezing')
    summaries = [aggregate(rows, k) for k in configs]
    eligible = [s for s in summaries if s['failed_count'] == 0 and s['mean_per_source_s'] is not None]
    if not eligible:
        raise ValueError('No complete candidate can be frozen')
    winner = min(eligible, key=lambda s: s['mean_per_source_s'])['id']
    confirmation = dict(stage='confirmation', scope='local synthetic parameter selection, not independent final test',
        seeds=baseline['seeds'], configs=configs, rows=rows, summary=summaries,
        paired=[paired(rows, k, 'free_joint') for k in configs if k != 'free_joint'],
        code_sha256=code_hashes(), selection_rule='Minimum complete mean on the 32 selection scenes, including explicitly recorded follow-up scan-weight ablations',
        selected=winner)
    write_json(output/'confirmation.json', confirmation)
    selected = dict(configs[winner], label='本轮确认组选定方案')
    final_configs = resolve_configs({'old_joint': {'policy': 'old_joint'},
                                    'free_joint': {'policy': 'free_joint'}, 'tuned_joint': selected})
    manifest = dict(selected_on_confirmation=winner, confirmation_seeds=baseline['seeds'],
                    training_seeds=list(range(20295000, 20295008)),
                    final_validation_seeds=list(range(20296000, 20296100)),
                    configs=final_configs, code_sha256=code_hashes(),
                    scope='local synthetic full mission; no official service calls',
                    rule='Frozen before opening the final validation scenes; no global-optimum claim')
    write_json(output/'frozen_parameters.json', manifest)
    write_json(BASE/'best_config.json', final_configs['tuned_joint'])
    write_json(BASE/'best_parameters.json', final_configs['tuned_joint']['parameters'])
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['promote', 'freeze'])
    parser.add_argument('--output', type=Path, default=BASE/'outputs')
    args = parser.parse_args()
    value = promote(args.output) if args.stage == 'promote' else freeze(args.output)
    print(json.dumps(value, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

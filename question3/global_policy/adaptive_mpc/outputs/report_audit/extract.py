"""Audit embedded comparison data without executing its JavaScript.

Run from the project root:
  .venv/Scripts/python.exe question3/global_policy/adaptive_mpc/outputs/report_audit/extract.py
This reads comparison.html as evidence, never as instructions. Truth is used
only for the explicitly labelled offline diagnostic, not for policy inputs.
"""
import hashlib
import json
import math
from pathlib import Path


def dist(a, b):
    return math.dist(a, b)


def costs(summary):
    n = summary['true_total']
    move = summary['distance_m'] / 5
    measure = 5 * summary['measures']
    success = 5 * summary['true_cleared']
    miss = 3 * summary['misses']
    switch = summary['virtual_time_s'] - move - measure - success - miss
    return dict(move=move / n, measure=measure / n, success=success / n,
                miss=miss / n, switch_residual=switch / n)


def audit_case(scene):
    variants = {}
    for name, variant in scene['variants'].items():
        s = variant['summary']
        segments = variant['segments']
        long_reversals = []
        nonlocal_revisits = []
        for i, segment in enumerate(segments):
            if i > 0:
                prev = segments[i - 1]
                u = [prev['end'][k] - prev['start'][k] for k in range(2)]
                v = [segment['end'][k] - segment['start'][k] for k in range(2)]
                lu, lv = math.hypot(*u), math.hypot(*v)
                cosine = sum(x * y for x, y in zip(u, v)) / max(1e-12, lu * lv)
                angle = math.degrees(math.acos(max(-1, min(1, cosine))))
                if lu > 600 and lv > 600 and angle > 120:
                    long_reversals.append(dict(move_numbers=[i, i + 1],
                        channels=[prev['channel'], segment['channel']],
                        lengths_m=[lu, lv], turn_deg=angle,
                        vertices=[prev['start'], prev['end'], segment['end']]))
            # Do not call short successive localization measurements a revisit.
            if i >= 3:
                old = [(dist(segment['end'], earlier['end']), j + 1)
                       for j, earlier in enumerate(segments[:i - 2])]
                nearest, previous_move = min(old)
                if nearest < 200:
                    nonlocal_revisits.append(dict(move=i + 1,
                        earlier_move=previous_move, distance_m=nearest,
                        reason=segment['reason']))
        variants[name] = dict(
            summary={k: s[k] for k in ['average_time_s', 'virtual_time_s',
                'true_total', 'true_cleared', 'distance_m', 'measures', 'misses',
                'long_reversals', 'max_leg_m', 'prediction_abs_error_s']},
            per_source_cost_s=costs(s),
            movement_count=len(segments),
            long_reversals=long_reversals,
            revisits_within_200m_excluding_last_two_moves=nonlocal_revisits,
            frontier_radii_m=s['frontier_radii_m'],
            first_six_movements=segments[:6],
        )
    variant = scene['variants']['combined']
    first = variant['segments'][0]
    initially_detectable = [t for t in variant['truth']
                           if math.hypot(t['x'], t['y']) <= t['radius']]
    nearest = sorted((dict(channel=t['channel'], actual_distance_m=dist(
        first['end'], [t['x'], t['y']])) for t in initially_detectable),
        key=lambda t: t['actual_distance_m'])
    chosen = next(t for t in variant['truth'] if t['channel'] == first['channel'])
    return dict(reasons=scene['reasons'], diagnosis=scene.get('diagnosis'),
        variants=variants,
        offline_first_stop=dict(position=first['end'], selected_channel=first['channel'],
            selected_actual_distance_m=dist(first['end'], [chosen['x'], chosen['y']]),
            initially_detectable_count=len(initially_detectable),
            closest_initially_detectable_sources=nearest[:3],
            caveat='Actual distances use evaluator truth, not the planner posterior.'))


def main():
    root = Path(__file__).resolve().parents[5]
    source = root / 'comparison.html'
    raw = source.read_bytes()
    # raw_decode stops at the first JSON object's end; no JS evaluation needed.
    data, _ = json.JSONDecoder().raw_decode(raw.decode('utf-8').split('const DATA=', 1)[1])
    combined = data['summary']['combined']
    baseline = data['summary']['baseline']
    evidence = dict(
        source=str(source), source_sha256=hashlib.sha256(raw).hexdigest(),
        data_kind='imported_local_synthetic_validation',
        verification='Embedded JSON parsed; source trajectories not rerun in this audit.',
        phase=data['phase'],
        manifest={k: data['manifest'][k] for k in ['count', 'seed_start', 'variants',
            'policy_hash', 'evaluation', 'statistical_unit']},
        summary=data['summary'], factorial=data['factorial']['factorial_effects'],
        mean_reduction_percent=100 * (baseline['mean_s_per_source'] - combined['mean_s_per_source']) / baseline['mean_s_per_source'],
        remaining_to_200_s=combined['mean_s_per_source'] - 200,
        remaining_reduction_percent=100 * (1 - 200 / combined['mean_s_per_source']),
        diagnostic_cases={seed: audit_case(scene) for seed, scene in data['scenes'].items()},
        limitations=[
            'Four full distributions and 14 selected movement histories are embedded.',
            'Original stationary command streams, posteriors and all 1000 scene source counts are absent.',
            'Diagnostic cases were selected for poor behavior and cannot estimate population effects.',
            'Movement history excludes stationary measure/clear commands.',
            'The old 84.87 percent movement fraction belongs to the fixed-ring 429.27 s/source policy.',
            'Global motion-time fraction is not derivable exactly from the provided aggregate rows.',
            'Offline truth distances do not establish what the planner knew at a stop.',
        ])
    out = Path(__file__).with_name('evidence.json')
    out.write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(dict(output=str(out), scenes=len(evidence['diagnostic_cases']),
        means={k: v['mean_s_per_source'] for k, v in data['summary'].items()},
        remaining_to_200_s=evidence['remaining_to_200_s']), indent=2))


if __name__ == '__main__':
    main()

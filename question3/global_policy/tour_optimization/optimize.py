"""Expensive black-box parameter search for the optimized tour.

The objective is the mean whole-scene seconds/source. Every proposal is run on
the same 14 development seeds (paired noise); a separate holdout is reserved.
Initial proposals use a Latin hypercube, followed by two deterministic
neighborhood proposals. This pilot is not Bayesian optimization. The later
bayesian_search module implements the actual GP and expected improvement.
"""
import argparse, json
from pathlib import Path
import numpy as np
from scipy.stats import qmc
from .experiment import batch

BOUNDS = {
    'early_unknown_scan_gain': (.04, .12),
    'mapping_radius_ratio': (.45, .75),
    'mapping_detection_probability': (.40, .70),
    'revisit_penalty_s': (0., 25.),
    'coupled_scan_weight': (0., .9),
}


def proposal_specs(count, seed=20260912):
    sampler = qmc.LatinHypercube(d=len(BOUNDS), seed=seed)
    x = sampler.random(count)
    keys = list(BOUNDS)
    specs = {}
    fixed = dict(refine_coverage=True, exact_route=True,
                 early_geometry_mapping=True, early_unknown_scan_gain=.08)
    for i, row in enumerate(x):
        fields = dict(fixed)
        for j, key in enumerate(keys):
            lo, hi = BOUNDS[key]
            fields[key] = float(lo + row[j] * (hi - lo))
        # Keep the search interpretable: the same field is used by the early
        # and normal unknown policy unless explicitly overridden by a proposal.
        fields['unknown_scan_gain'] = fields['early_unknown_scan_gain']
        specs[f'proposal_{i:02d}'] = fields
    return specs


def main():
    p=argparse.ArgumentParser();p.add_argument('--initial',type=int,default=8);p.add_argument('--seeds',type=int,default=14)
    p.add_argument('--seed-start',type=int,default=20273000);p.add_argument('--workers',type=int,default=10)
    p.add_argument('--output',required=True);a=p.parse_args()
    if a.initial<4: p.error('initial must be >=4')
    out=Path(a.output)
    if out.exists() and any(out.iterdir()):raise ValueError('Choose a fresh output directory')
    out.mkdir(parents=True)
    specs=proposal_specs(a.initial)
    seeds=list(range(a.seed_start,a.seed_start+a.seeds))
    rows,stats=batch(specs,seeds,out,a.workers)
    history=[]
    for name,s in stats.items(): history.append(dict(name=name,score=s['objective'],fields=specs[name]))
    history.sort(key=lambda z:z['score'])
    # Pilot only: perturb both best points in opposite coordinate directions.
    best=history[:2]; extra={}
    for i,item in enumerate(best):
        fields=dict(item['fields'])
        for key,(lo,hi) in BOUNDS.items():
            center=fields[key]; span=(hi-lo)*.16
            fields[key]=float(np.clip(center + (i*2-1)*span,lo,hi))
        fields['unknown_scan_gain']=fields['early_unknown_scan_gain']
        extra[f'local_{i:02d}']=fields
    if extra:
        r2,s2=batch(extra,seeds,out,a.workers);rows.update(r2);stats.update(s2)
        history += [dict(name=name,score=s['objective'],fields=extra[name]) for name,s in s2.items()]
    (out/'cases.json').write_text(json.dumps(rows),encoding='utf8')
    (out/'summary.json').write_text(json.dumps(stats,indent=2),encoding='utf8')
    history.sort(key=lambda z:z['score'])
    report=dict(algorithm='Latin hypercube + two deterministic neighborhood candidates',
                bounds=BOUNDS,development_seeds=seeds,proposals=len(history),ranking=history,
                best=history[0],objective='mean whole-scene seconds/source; incomplete=penalty')
    (out/'optimization.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()

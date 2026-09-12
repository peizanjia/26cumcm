"""GP expected-improvement proposals + progressively larger paired seed blocks.

This uses a GP surrogate and successive promotion, not the BOHB KDE algorithm.
Scenario count changes the evaluation fidelity; physical parameters, posterior
sample budgets and completion certificates remain fixed throughout the race.
"""
from .experiment import batch,freeze,environment
import argparse,json,hashlib
from pathlib import Path
from dataclasses import asdict
import numpy as np
from scipy.stats import qmc
from .parameters import Parameters
from .gaussian_process import GaussianProcess,expected_improvement


BOUNDS={
    'unknown_scan_gain':(.04,.24),
    'mapping_radius_ratio':(.4,.85),
    'mapping_detection_probability':(.35,.8),
    'coupled_scan_weight':(0.,1.),
    'initial_step':(300.,900.),
    'early_geometry_stops':(3.,10.),
    'revisit_penalty_s':(0.,25.),
}
BASE=dict(refine_coverage=True,exact_route=True)
DEFAULTS=asdict(Parameters(**BASE))


def vector(fields):
    return np.array([(fields.get(k,DEFAULTS[k])-lo)/(hi-lo) for k,(lo,hi) in BOUNDS.items()])


def fields_from_vector(x):
    fields=dict(BASE)
    for value,(key,(lo,hi)) in zip(x,BOUNDS.items()):
        fields[key]=float(lo+(hi-lo)*np.clip(value,0,1))
    fields['early_geometry_stops']=int(round(fields['early_geometry_stops']))
    fields['early_unknown_scan_gain']=fields['unknown_scan_gain']
    Parameters(**fields).validate()
    return fields


def scores(rows,base):
    baseline={r['seed']:r for r in base}
    loss=lambda r:r['average_time_s'] if r['complete'] else 1e6
    paired=np.array([loss(row)-loss(baseline[row['seed']]) for row in rows])
    return dict(objective=float(np.mean([loss(row) for row in rows])),
                delta=float(paired.mean()),variance=float(paired.var(ddof=1)/len(paired)) if len(paired)>1 else 0.,
                complete=all(r['complete'] for r in rows),count=len(rows))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pilot',default='question3/global_policy/tour_optimization/outputs/parameter_search')
    p.add_argument('--output',required=True);p.add_argument('--workers',type=int,default=10)
    p.add_argument('--iterations',type=int,default=12)
    a=p.parse_args();out=Path(a.output)
    if out.exists() and any(out.iterdir()):raise ValueError('Use a new output directory')
    out.mkdir(parents=True)
    seed_blocks=[list(range(20273200,20273214)),list(range(20273400,20273414)),list(range(20273500,20273514))]
    confirm=list(range(20273600,20273642))
    pilot=Path(a.pilot);original=json.loads((pilot/'optimization.json').read_text(encoding='utf8'))
    specs={'structural':dict(BASE)}
    for row in original['ranking']:specs['pilot_'+row['name']]=row['fields']
    initial=qmc.LatinHypercube(d=len(BOUNDS),seed=8041).random(5)
    for i,x in enumerate(initial):specs[f'initial_{i:02d}']=fields_from_vector(x)
    manifest=dict(algorithm='Matern52_GP_expected_improvement_with_paired_noise_and_successive_promotion',
                  bounds=BOUNDS,seed_blocks=seed_blocks,confirmation_seeds=confirm,
                  iterations=a.iterations,initial_specs=specs,environment=environment(),
                  source_hash=freeze(out),pilot_sha256=hashlib.sha256((pilot/'optimization.json').read_bytes()).hexdigest(),
                  evaluation='local_synthetic_parameter_selection',
                  references=['https://arxiv.org/abs/1206.2944','https://proceedings.mlr.press/v80/falkner18a.html'])
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
    # Reuse identical full-precision pilot scene outputs, without calling a
    # simulator again. Candidate parameters are checked by batch before reuse.
    for old in original['ranking']:
        name='pilot_'+old['name'];folder=out/'low'/name;folder.mkdir(parents=True)
        for seed in seed_blocks[0]:
            row=json.loads((pilot/old['name']/f'{seed}.summary.json').read_text(encoding='utf8'))
            row['variant']=name
            (folder/f'{seed}.summary.json').write_text(json.dumps(row),encoding='utf8')
    data,_=batch(specs,seed_blocks[0],out/'low',a.workers,trace=False)
    observations={name:scores(items,data['structural']) for name,items in data.items()}
    suggestions=[];rng=np.random.default_rng(6733)
    for iteration in range(a.iterations):
        names=list(specs);X=np.array([vector(specs[name]) for name in names])
        y=np.array([observations[name]['delta'] for name in names])
        variance=np.array([observations[name]['variance'] for name in names])
        gp=GaussianProcess(X,y,variance)
        pool=qmc.Sobol(d=len(BOUNDS),scramble=True,seed=9410+iteration).random_base2(11)
        best=names[int(np.argmin(y))]
        near=np.clip(vector(specs[best])+rng.normal(0,.13,(512,len(BOUNDS))),0,1)
        pool=np.vstack((pool,near))
        # Round the integer dimension before scoring and duplicate avoidance.
        candidates=[fields_from_vector(x) for x in pool]
        pool=np.array([vector(fields) for fields in candidates])
        mu,sigma=gp.predict(pool)
        incumbent=float(np.min(gp.predict(X)[0]))
        ei=expected_improvement(mu,sigma,incumbent)
        distance=np.min(np.linalg.norm(pool[:,None]-X[None,:],axis=2),axis=1)
        ei[distance<1e-4]=-np.inf
        index=int(np.argmax(ei));name=f'bo_{iteration:02d}';specs[name]=candidates[index]
        suggestion=dict(name=name,fields=specs[name],predicted_delta=float(mu[index]),
                        predicted_std=float(sigma[index]),ei=float(ei[index]),length_scale=gp.length)
        suggestions.append(suggestion)
        (out/'suggestions.json').write_text(json.dumps(suggestions,indent=2),encoding='utf8')
        new,_=batch({name:specs[name]},seed_blocks[0],out/name,a.workers,trace=False)
        data[name]=new[name];observations[name]=scores(new[name],data['structural'])
        print(json.dumps(dict(iteration=iteration,name=name,**observations[name],best_delta=min(o['delta'] for o in observations.values()))),flush=True)
        (out/'low_observations.json').write_text(json.dumps(observations,indent=2),encoding='utf8')
    ordered=sorted(specs,key=lambda n:observations[n]['objective'])
    promoted=list(dict.fromkeys(['structural']+ordered[:5]))
    later=seed_blocks[1]+seed_blocks[2]
    medium,_=batch({n:specs[n] for n in promoted},later,out/'medium',a.workers,trace=False)
    combined={n:data[n]+medium[n] for n in promoted}
    medium_scores={n:scores(items,combined['structural']) for n,items in combined.items()}
    finalists=list(dict.fromkeys(['structural']+sorted(promoted,key=lambda n:medium_scores[n]['objective'])[:2]))
    high,_=batch({n:specs[n] for n in finalists},confirm,out/'confirmation',a.workers,trace=True)
    complete={n:combined[n]+high[n] for n in finalists}
    high_scores={n:scores(items,complete['structural']) for n,items in complete.items()}
    winner=min(finalists,key=lambda n:high_scores[n]['objective'])
    result=dict(algorithm=manifest['algorithm'],all_specs=specs,low=observations,
                medium=medium_scores,high=high_scores,seed_blocks=seed_blocks,confirmation_seeds=confirm,
                promoted=promoted,finalists=finalists,best_name=winner,best_fields=specs[winner],
                limitation='Best tested selection candidate; not a proven global optimum. Final test seeds remain unused.')
    (out/'optimization.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    (out/'best_parameters.json').write_text(json.dumps(asdict(Parameters(**specs[winner])),indent=2),encoding='utf8')
    (out/'all_cases.json').write_text(json.dumps(dict(low=data,medium=medium,confirmation=high)),encoding='utf8')
    print(json.dumps(dict(best_name=winner,best_fields=specs[winner],high=high_scores),indent=2),flush=True)


if __name__=='__main__':main()

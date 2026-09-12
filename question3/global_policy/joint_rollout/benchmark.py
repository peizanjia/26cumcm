"""Resumable paired scenario evaluation, multiprocessing, full compressed traces."""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[_key]='1'
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from dataclasses import asdict
import gzip,hashlib,json,math,time
from pathlib import Path
import numpy as np
from scipy.stats import t as student_t,ttest_1samp,beta
from .parameters import JointParameters
from .planner import run_local
from .run import VARIANTS


def policy_hash():
    root=Path(__file__).resolve().parents[2];files=[]
    for name in ('global_policy/dynamic','global_policy/route_study'):
        files+=list((root/name).glob('*.py'))
    files += [root/'global_policy'/n for n in ('model.py','decisions.py','runner.py')]
    files += [Path(__file__).with_name(n) for n in ('planner.py','parameters.py','service.py','kernel.py')]
    files += [root/'local_sim/simulator.py',root.parent/'question1/geometry.py',root.parent/'question1/enclosing_circle.py']
    h=hashlib.sha256()
    for path in sorted(set(files)):
        h.update(str(path).encode());h.update(path.read_bytes())
    return h.hexdigest()


def metrics(planner,row):
    q=np.zeros(2);moves=[];lengths=[];sectors=[];last=0
    for c in planner.world.commands:
        if c['position'] is None:continue
        y=np.array(c['position']);d=y-q;length=float(np.linalg.norm(d))
        if length>1e-7:moves.append(d);lengths.append(length)
        q=y
    folds=0
    for a,b in zip(moves,moves[1:]):
        la,lb=np.linalg.norm(a),np.linalg.norm(b)
        if min(la,lb)>600 and np.dot(a,b)/(la*lb)<-.5:folds+=1
    estimates=[];residuals=[]
    commands=planner.world.commands
    for e in planner.events:
        if e['phase']!='local_decision' or not e.get('candidates'):continue
        c=next((v for v in e['candidates'] if v.get('selected')),None)
        if c is None:continue
        i=e['command_index'];start=commands[i-1]['response']['virtual_time_s'] if i else 0
        end=next((a['response']['virtual_time_s'] for a in commands[i:]
                  if a['channel']==e['channel'] and a['response'].get('clear_result')=='success'),None)
        if end is not None:residuals.append(end-start-c['expected_remaining_s'])
        estimates.append(c)
    row.update(long_reversals=folds,max_leg_m=max(lengths,default=0),
        prediction_residual_mean_s=float(np.mean(residuals)) if residuals else None,
        prediction_abs_error_s=float(np.mean(np.abs(residuals))) if residuals else None,
        candidates_evaluated=sum(v.get('candidate_evaluations',0) for v in estimates),
        max_rollout_guard_probability=max((v.get('guard_probability',0) for v in estimates),default=0))
    return row


def task(args):
    seed,name,folder,overrides=args
    params=JointParameters(**{**VARIANTS[name],**overrides})
    row,planner=run_local(seed,params);row=metrics(planner,row);row['variant']=name
    # Truth is attached by evaluator only after planner exits.
    truth=[asdict(s) for s in planner.client.simulator.sources]
    pack=dict(summary=row,commands=planner.world.commands,events=planner.events,truth=truth)
    folder=Path(folder)/name;folder.mkdir(parents=True,exist_ok=True)
    filename=folder/f'{seed}.json.gz'
    with gzip.open(filename.with_suffix('.tmp'),'wt',encoding='utf8',compresslevel=3) as f:json.dump(pack,f,ensure_ascii=False)
    filename.with_suffix('.tmp').replace(filename)
    (folder/f'{seed}.summary.json').write_text(json.dumps(row),encoding='utf8')
    return row


def summarize(rows):
    lookup={name:{r['seed']:r for r in items} for name,items in rows.items()}
    control=lookup['baseline'];report={};pvalues=[]
    for name,items in rows.items():
        complete=[r for r in items if r['complete']];n=len(items);bad=n-len(complete)
        v=np.array([r['average_time_s'] for r in complete]);result=dict(count=n,complete=len(complete),
            failures=bad,failure_upper95=float(beta.ppf(.95,bad+1,n-bad)) if bad<n else 1.,
            true_cleared=sum(r['true_cleared'] for r in items),
            mean_s_per_source=float(v.mean()) if len(v) else None,
            timing_conditional_on_completion=bool(bad),
            p50=float(np.median(v)) if len(v) else None,p90=float(np.quantile(v,.9)) if len(v) else None,
            p95=float(np.quantile(v,.95)) if len(v) else None,worst=float(v.max()) if len(v) else None,
            cvar95=float(v[v>=np.quantile(v,.95)].mean()) if len(v) else None,
            mean_distance_m=float(np.mean([r['distance_m'] for r in items])),
            mean_measures=float(np.mean([r['measures'] for r in items])),misses=sum(r['misses'] for r in items),
            long_reversals=sum(r['long_reversals'] for r in items),
            mean_wall_s=float(np.mean([r['wall_time_s'] for r in items])))
        paired=[r for r in items if r['seed'] in control]
        if len(paired)>1 and all(r['complete'] and control[r['seed']]['complete'] for r in paired):
            d=np.array([r['average_time_s']-control[r['seed']]['average_time_s'] for r in paired])
            se=float(d.std(ddof=1)/math.sqrt(len(d)));half=float(student_t.ppf(.975,len(d)-1)*se)
            rng=np.random.default_rng(714902);boot=np.array([rng.choice(d,len(d),replace=True).mean() for _ in range(2000)])
            result.update(paired_n=len(d),paired_delta_s=float(d.mean()),paired_se_s=se,
                paired_ci95=[float(d.mean()-half),float(d.mean()+half)],
                paired_bootstrap_ci95=np.quantile(boot,[.025,.975]).tolist(),faster=int((d<0).sum()))
            if name!='baseline':pvalues.append((float(ttest_1samp(d,0).pvalue) if se else 1.,name))
        report[name]=result
    previous=0.
    for i,(p,name) in enumerate(sorted(pvalues)):
        previous=max(previous,min(1.,p*(len(pvalues)-i)));report[name]['holm_p']=previous
    return report


def main():
    p=argparse.ArgumentParser();p.add_argument('--count',type=int,default=1000);p.add_argument('--seed-start',type=int,default=20271000)
    p.add_argument('--workers',type=int,default=10);p.add_argument('--variants',nargs='+',choices=list(VARIANTS),default=list(VARIANTS))
    p.add_argument('--output',default='question3/global_policy/joint_rollout/outputs/validation');p.add_argument('--params')
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    overrides=json.loads(Path(a.params).read_text(encoding='utf8')) if a.params else {}
    names=list(dict.fromkeys(['baseline']+a.variants));seeds=list(range(a.seed_start,a.seed_start+a.count))
    config=dict(count=a.count,seed_start=a.seed_start,variants=names,overrides=overrides,policy_hash=policy_hash(),
                evaluation='local_synthetic_paired',statistical_unit='whole_scene',
                parameters={name:asdict(JointParameters(**{**VARIANTS[name],**overrides})) for name in names})
    configfile=out/'manifest.json'
    if configfile.exists() and json.loads(configfile.read_text())!=config:raise RuntimeError('Manifest changed; use a new output directory')
    configfile.write_text(json.dumps(config,indent=2),encoding='utf8')
    rows={name:[] for name in names};pending=[]
    for seed in seeds:
        for name in names:
            file=out/name/f'{seed}.summary.json'
            if file.exists():rows[name].append(json.loads(file.read_text()))
            else:pending.append((seed,name,str(out),overrides))
    done=sum(map(len,rows.values()));started=time.perf_counter()
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures=[pool.submit(task,x) for x in pending]
        for future in as_completed(futures):
            row=future.result();rows[row['variant']].append(row);done+=1
            if done%20==0 or not row['complete']:
                print(json.dumps(dict(done=done,total=len(seeds)*len(names),elapsed_s=round(time.perf_counter()-started),
                    last_variant=row['variant'],last_seed=row['seed'],complete=row['complete'],failure=row['failure'])),flush=True)
    for items in rows.values():items.sort(key=lambda r:r['seed'])
    report=summarize(rows)
    (out/'cases.json').write_text(json.dumps(rows),encoding='utf8')
    (out/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    print(json.dumps(report,indent=2),flush=True)
    if not all(r['complete'] for items in rows.values() for r in items):raise SystemExit(1)


if __name__=='__main__':main()

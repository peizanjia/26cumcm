"""Reproducible paired LOCAL simulations with whole-scene statistics."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[name]='1'
import argparse,gzip,hashlib,json,platform,time,zipfile
from importlib.metadata import version
from concurrent.futures import ProcessPoolExecutor,as_completed
from dataclasses import asdict
from pathlib import Path
import numpy as np
from scipy.stats import t as student_t
from .parameters import AdaptiveParameters
from .planner import run_local

VARIANTS={'adaptive':{},'no_stop_scans':{'stop_scans':False,'side_scan_channels':0},
          'no_revisit':{'revisit_penalty_s':0.},'no_early':{'early_mapping':False},
          'lean':{'unknown_scan_gain':.08,'early_unknown_scan_gain':.035},
          'lean_no_early':{'unknown_scan_gain':.08,'early_unknown_scan_gain':.035,'early_mapping':False},
          'lean_no_revisit':{'unknown_scan_gain':.08,'early_unknown_scan_gain':.035,'revisit_penalty_s':0.},
          'lean_no_waypoints':{'unknown_scan_gain':.08,'early_unknown_scan_gain':.035,'max_waypoint_step':1800.},
          'anticipated':{'anticipate_coverage':True,'unknown_scan_gain':.08,'early_unknown_scan_gain':.035},
          'anticipated_frequent':{'anticipate_coverage':True},
          'map_global':{'unknown_scan_gain':.08,'early_unknown_scan_gain':.035,'early_geometry_mapping':True},
          'sweep80':{'unknown_scan_gain':.08,'early_unknown_scan_gain':.035,'early_geometry_mapping':True,'angular_bias_s':80.},
          'sweep160':{'unknown_scan_gain':.08,'early_unknown_scan_gain':.035,'early_geometry_mapping':True,'angular_bias_s':160.},
          'sweep_candidate':{'engine':'sweep','early_geometry_mapping':True,'early_unknown_scan_gain':.08,'revisit_penalty_s':15.},
          'tour_candidate':{'engine':'tour','early_geometry_mapping':True,'early_unknown_scan_gain':.08,'revisit_penalty_s':15.}}

def source_files():
    root=Path(__file__).resolve().parents[2]
    paths=[root/'global_policy'/s for s in ('model.py','decisions.py','runner.py')]
    for folder in ('dynamic','route_study'):
        paths+=list((root/'global_policy'/folder).glob('*.py'))
    paths += [root/'global_policy/joint_rollout'/s for s in ('planner.py','parameters.py','service.py','kernel.py')]
    paths += [root/'global_policy/adaptive_mpc'/s for s in ('planner.py','parameters.py','planning.py','scanning.py','exploration.py','anticipated_tail.py','sweep_planner.py','tour_planner.py','benchmark.py')]
    paths += [root/'local_sim/simulator.py',root.parent/'question1/geometry.py',root.parent/'question1/enclosing_circle.py']
    return [f for f in sorted(set(paths)) if 'outputs' not in f.parts and not f.name.startswith('test_')]

def source_hash():
    root=Path(__file__).resolve().parents[2];h=hashlib.sha256()
    for f in source_files():
        h.update(str(f.relative_to(root.parent)).encode());h.update(f.read_bytes())
    return h.hexdigest()

def task(args):
    seed,variant,folder=args
    if variant=='combined':
        from ..joint_rollout.planner import run_local as old_run
        row,planner=old_run(seed)
    else:
        fields=VARIANTS[variant].copy();engine=fields.pop('engine','global')
        if engine in ('sweep','tour'):
            from .sweep_planner import SweepParameters
            if engine=='sweep':from .sweep_planner import run_local as selected_run
            else:from .tour_planner import run_local as selected_run
            row,planner=selected_run(seed,SweepParameters(**fields))
        else:row,planner=run_local(seed,AdaptiveParameters(**fields))
    from ..joint_rollout.benchmark import metrics
    row=metrics(planner,row);row['variant']=variant
    components=dict(movement=planner.world.distance/5,measure=0,switch=0,clear=0);channel=1
    for c in planner.world.commands:
        if c['path']=='/measure':
            components['measure']+=5;components['switch']+=int(channel!=c['channel']);channel=c['channel']
        if c['path']=='/clear':components['clear']+=5 if c['response']['clear_result']=='success' else 3
    row['time_components_s']=components
    pack=dict(summary=row,commands=planner.world.commands,events=planner.events,
              truth=[asdict(s) for s in planner.client.simulator.sources])
    if hasattr(planner,'memory'):pack['exploration_memory']=planner.memory.snapshot()
    folder=Path(folder)/variant;folder.mkdir(parents=True,exist_ok=True)
    with gzip.open(folder/f'{seed}.json.gz','wt',encoding='utf8',compresslevel=3) as f:json.dump(pack,f,ensure_ascii=False)
    (folder/f'{seed}.summary.json').write_text(json.dumps(row),encoding='utf8')
    return row

def summarize(rows):
    control={r['seed']:r for r in rows['combined']};result={}
    for variant,items in rows.items():
        ok=[r for r in items if r['complete']];n=len(items)
        v=np.array([r['average_time_s'] for r in ok])
        row=dict(count=n,complete=len(ok),failures=n-len(ok),true_cleared=sum(r['true_cleared'] for r in items),
            mean_s_per_source=float(v.mean()) if len(v) else None,
            timing_conditional_on_completion=len(ok)<n,
            p90=float(np.quantile(v,.9)) if len(v) else None,p95=float(np.quantile(v,.95)) if len(v) else None,
            worst=float(v.max()) if len(v) else None,mean_distance_m=float(np.mean([r['distance_m'] for r in items])),
            mean_measures=float(np.mean([r['measures'] for r in items])),mean_wall_s=float(np.mean([r['wall_time_s'] for r in items])),
            long_reversals=sum(r['long_reversals'] for r in items),
            failure_upper95_if_zero=float(1-.05**(1/n)) if n==len(ok) else None,
            mean_components_s={k:float(np.mean([r['time_components_s'][k] for r in items])) for k in ('movement','measure','switch','clear')})
        paired=[r for r in items if r['complete'] and control.get(r['seed'],{}).get('complete')]
        if len(paired)>1:
            d=np.array([r['average_time_s']-control[r['seed']]['average_time_s'] for r in paired])
            half=float(student_t.ppf(.975,len(d)-1)*d.std(ddof=1)/np.sqrt(len(d)))
            row.update(paired_n=len(d),paired_delta_s=float(d.mean()),paired_ci95=[float(d.mean()-half),float(d.mean()+half)],
                       faster=int((d<0).sum()))
        result[variant]=row
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--count',type=int,default=20);p.add_argument('--seed-start',type=int,default=20272300)
    p.add_argument('--seeds',nargs='+',type=int);p.add_argument('--workers',type=int,default=8)
    p.add_argument('--variants',nargs='+',choices=list(VARIANTS),default=['adaptive'])
    p.add_argument('--output',default='question3/global_policy/adaptive_mpc/outputs/development')
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    seeds=a.seeds or list(range(a.seed_start,a.seed_start+a.count));variants=['combined']+list(dict.fromkeys(a.variants))
    manifest=dict(seeds=seeds,variants=variants,variant_overrides={v:VARIANTS.get(v,{}) for v in variants},source_hash=source_hash(),evaluation='local_synthetic_paired',python=platform.python_version(),
                  statistical_unit='whole_scene',adaptive_parameters=asdict(AdaptiveParameters()),
                  packages={name:version(name) for name in ('numpy','scipy','numba','networkx')})
    path=out/'manifest.json'
    if path.exists() and json.loads(path.read_text(encoding='utf8'))!=manifest:raise RuntimeError('Use a fresh output for changed code/config')
    path.write_text(json.dumps(manifest,indent=2),encoding='utf8')
    with zipfile.ZipFile(out/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as snapshot:
        root=Path(__file__).resolve().parents[3]
        for source in source_files():snapshot.write(source,source.relative_to(root).as_posix())
    rows={v:[] for v in variants};pending=[]
    for seed in seeds:
        for v in variants:
            f=out/v/f'{seed}.summary.json'
            if f.exists():rows[v].append(json.loads(f.read_text(encoding='utf8')))
            else:pending.append((seed,v,str(out)))
    start=time.perf_counter();done=sum(map(len,rows.values()))
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for future in as_completed([pool.submit(task,arg) for arg in pending]):
            r=future.result();rows[r['variant']].append(r);done+=1
            if done%10==0 or not r['complete']:print(json.dumps(dict(done=done,total=len(seeds)*len(variants),elapsed_s=round(time.perf_counter()-start),seed=r['seed'],variant=r['variant'],complete=r['complete'],failure=r['failure'])),flush=True)
    for items in rows.values():items.sort(key=lambda r:r['seed'])
    report=summarize(rows)
    (out/'cases.json').write_text(json.dumps(rows),encoding='utf8')
    (out/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    print(json.dumps(report,indent=2),flush=True)
    if any(not r['complete'] for v in rows.values() for r in v):raise SystemExit(1)

if __name__=='__main__':main()

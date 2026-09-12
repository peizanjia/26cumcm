"""Cached, paired local experiments. Failed cases remain in the objective."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[key]='1'
import argparse,gzip,hashlib,json,time,zipfile
from concurrent.futures import ProcessPoolExecutor,as_completed
from dataclasses import asdict
from pathlib import Path
import platform
from importlib.metadata import version
import numpy as np
from scipy.stats import t


def source_files():
    from ..adaptive_mpc.benchmark import source_files as previous
    return sorted(set(previous()+[p for p in Path(__file__).parent.glob('*.py') if not p.name.startswith('test_')]))


def freeze(folder):
    root=Path(__file__).resolve().parents[3];h=hashlib.sha256();folder=Path(folder)
    with zipfile.ZipFile(folder/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in source_files():
            rel=p.relative_to(root).as_posix();data=p.read_bytes()
            h.update(rel.encode());h.update(data);z.writestr(rel,data)
    return h.hexdigest()


def environment():
    return dict(python=platform.python_version(),packages={key:version(key) for key in ('numpy','scipy','numba')})


def task(seed,name,spec,folder,trace=True):
    fields={k:v for k,v in spec.items() if k!='engine'}
    if spec.get('engine')=='previous':
        from ..adaptive_mpc.tour_planner import run_local
        from ..adaptive_mpc.sweep_planner import SweepParameters
        defaults=dict(early_geometry_mapping=True,early_unknown_scan_gain=.08,revisit_penalty_s=15.)
        defaults.update(fields);parameters=SweepParameters(**defaults)
    else:
        from .planner import run_local
        from .parameters import Parameters
        parameters=Parameters(**fields)
    row,planner=run_local(seed,parameters)
    row['variant']=name
    from ..joint_rollout.benchmark import metrics
    row=metrics(planner,row)
    out=Path(folder)/name;out.mkdir(parents=True,exist_ok=True)
    if trace:
        package=dict(summary=row,commands=planner.world.commands,events=planner.events,
                     truth=[asdict(s) for s in planner.client.simulator.sources])
        with gzip.open(out/f'{seed}.json.gz','wt',encoding='utf8',compresslevel=3) as f:
            json.dump(package,f,ensure_ascii=False)
    (out/f'{seed}.summary.json').write_text(json.dumps(row,ensure_ascii=False),encoding='utf8')
    return row


def batch(specs,seeds,folder,workers=10,trace=True):
    out=Path(folder);out.mkdir(parents=True,exist_ok=True)
    rows={name:[] for name in specs};pending=[]
    for name,spec in specs.items():
        for seed in seeds:
            path=out/name/f'{seed}.summary.json'
            if path.exists():
                row=json.loads(path.read_text(encoding='utf8'))
                for key,value in spec.items():
                    if key!='engine' and row['parameters'].get(key)!=value:
                        raise ValueError(f'Cached parameter mismatch: {name}/{seed}/{key}')
                rows[name].append(row)
            else:pending.append((seed,name,spec,str(out),trace))
    start=time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(task,*args) for args in pending]
        for done,f in enumerate(as_completed(futures),1):
            row=f.result();rows[row['variant']].append(row)
            if done%10==0 or not row['complete']:
                print(json.dumps(dict(done=done,total=len(pending),elapsed_s=round(time.perf_counter()-start),
                                     variant=row['variant'],seed=row['seed'],complete=row['complete'],failure=row['failure'])),flush=True)
    for items in rows.values():items.sort(key=lambda r:r['seed'])
    (out/'cases.json').write_text(json.dumps(rows,ensure_ascii=False),encoding='utf8')
    result=summarize(rows)
    (out/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    return rows,result


def summarize(rows):
    first=next(iter(rows));base={r['seed']:r for r in rows[first]};result={}
    for name,items in rows.items():
        scores=np.array([r['average_time_s'] if r['complete'] else 1e6 for r in items])
        valid=[r for r in items if r['complete']]
        d=np.array([r['average_time_s']-base[r['seed']]['average_time_s'] for r in valid if base[r['seed']]['complete']])
        half=float(t.ppf(.975,len(d)-1)*d.std(ddof=1)/np.sqrt(len(d))) if len(d)>1 else None
        result[name]=dict(count=len(items),complete=len(valid),objective=float(scores.mean()),
                         mean_s_per_source=float(np.mean([r['average_time_s'] for r in valid])) if valid else None,
                         paired_delta_s=float(d.mean()) if len(d) else None,
                         paired_ci95=[float(d.mean()-half),float(d.mean()+half)] if half is not None else None,
                         faster=int((d<0).sum()),true_cleared=sum(r['true_cleared'] for r in items),
                         mean_wall_s=float(np.mean([r['wall_time_s'] for r in items])))
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--specs',required=True);p.add_argument('--seeds',nargs='+',type=int)
    p.add_argument('--count',type=int,default=14);p.add_argument('--seed-start',type=int,default=20273000)
    p.add_argument('--workers',type=int,default=10);p.add_argument('--output',required=True)
    p.add_argument('--evaluation',choices=['development','validation'],default='development')
    a=p.parse_args();specs=json.loads(Path(a.specs).read_text(encoding='utf8'));seeds=a.seeds or list(range(a.seed_start,a.seed_start+a.count))
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    if (out/'manifest.json').exists():raise ValueError('Choose a new output directory')
    from .parameters import Parameters
    for spec in specs.values():Parameters(**{k:v for k,v in spec.items() if k!='engine'}).validate()
    manifest=dict(specs=specs,seeds=seeds,evaluation='local_synthetic_'+a.evaluation,source_hash=freeze(out),environment=environment())
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
    _,result=batch(specs,seeds,out,a.workers)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()

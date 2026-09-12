"""Paired local-only algorithm ablations. Saved controls never use hidden planner data."""
import argparse,json
from dataclasses import asdict,replace
from pathlib import Path
import numpy as np
from .planner import run_local
from .parameters import StudyParameters


VARIANTS={
    'baseline':{},
    'nearest':dict(route_method='nearest'),
    'exact_dp':dict(route_method='exact_dp'),
    'inner800':dict(global_mode='inner_first',inner_radius=800.),
    'inner1100':dict(global_mode='inner_first',inner_radius=1100.),
    'adaptive_sector':dict(global_mode='adaptive_sector'),
    'voi_stops':dict(scan_mode='voi'),
    'voi_each_stop':dict(scan_mode='voi',scan_each_probe=True),
    'nearest_voi':dict(route_method='nearest',scan_mode='voi',scan_each_probe=True),
    'exact_open':dict(route_method='exact_open'),
    'mst_2opt':dict(route_method='mst_2opt'),
    'christofides':dict(route_method='christofides'),
    'annealing':dict(route_method='annealing'),
    'block_first':dict(global_mode='block_first',route_method='exact_open'),
    'global_guided':dict(route_scope='global_guided',route_method='exact_open'),
    'global_graph':dict(global_mode='global_graph',route_method='exact_open'),
    'endpoint_aware':dict(route_method='exact_dp',open_if_no_frontier=True),
}


def aggregate(rows,control):
    if not all(r['complete'] for r in rows):return dict(complete=sum(r['complete'] for r in rows),failure='Incomplete cases; no completion-time comparison')
    values=np.array([r['average_time_s'] for r in rows]);reference=np.array([r['average_time_s'] for r in control]);delta=values-reference
    sizes=[n for r in rows for n in r.get('route_sizes',[])]
    return dict(count=len(rows),complete=len(rows),true_cleared=sum(r['true_cleared'] for r in rows),
                mean_s_per_source=float(values.mean()),paired_difference_s=float(delta.mean()),
                paired_se_s=float(delta.std(ddof=1)/np.sqrt(len(delta))) if len(delta)>1 else None,
                faster_cases=int((delta<-1e-6).sum()),mean_distance_m=float(np.mean([r['distance_m'] for r in rows])),
                mean_measures=float(np.mean([r['measures'] for r in rows])),misses=sum(r['misses'] for r in rows),
                mean_wall_s=float(np.mean([r['wall_time_s'] for r in rows])),
                mean_voi_scans=float(np.mean([r.get('voi_scans',0) for r in rows])),
                route_calls=len(sizes) if sizes else None,multi_target_route_calls=sum(n>1 for n in sizes) if sizes else None,
                max_route_size=max(sizes) if sizes else None)


def main():
    p=argparse.ArgumentParser();p.add_argument('--count',type=int,default=20);p.add_argument('--seed-start',type=int,default=20263000)
    p.add_argument('--variants',nargs='+',choices=list(VARIANTS),default=['baseline','nearest','exact_dp','inner800','inner1100','adaptive_sector','voi_stops','voi_each_stop']);p.add_argument('--reuse-baseline',action='store_true')
    p.add_argument('--output',default='question3/global_policy/route_study/outputs/screening');a=p.parse_args()
    if a.count<1:p.error('count must be positive')
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    seeds=list(range(a.seed_start,a.seed_start+a.count));cases={};summaries={}
    names=list(dict.fromkeys(['baseline']+a.variants))
    for name in names:
        params=StudyParameters(**VARIANTS[name]);rows=[]
        if name=='baseline' and a.reuse_baseline:
            old=json.loads(Path('question3/global_policy/dynamic/outputs/final_validation/cases.json').read_text(encoding='utf8'))['dynamic']
            lookup={r['seed']:r for r in old}
            if not all(s in lookup for s in seeds):p.error('Requested control seeds are not in saved baseline')
            rows=[lookup[s] for s in seeds]
        else:
            with (out/f'{name}.jsonl').open('w',encoding='utf8') as log:
                for i,seed in enumerate(seeds):
                    row,_=run_local(seed,params,out/name/f'seed_{seed}' if i<2 else None)
                    rows.append(row);log.write(json.dumps(row)+'\n');log.flush()
                    if (i+1)%5==0 or not row['complete']:print(name,i+1,'/',len(seeds),'complete',row['complete'],'avg',row['average_time_s'],row['failure'],flush=True)
        cases[name]=rows;summaries[name]=aggregate(rows,cases['baseline'])
        print('RESULT',name,json.dumps(summaries[name]),flush=True)
        (out/'cases.json').write_text(json.dumps(cases,indent=2),encoding='utf8')
        (out/'summary.json').write_text(json.dumps(dict(evaluation='local_paired_algorithm_comparison',seed_start=a.seed_start,count=a.count,
            variants=summaries,parameters={n:asdict(StudyParameters(**VARIANTS[n])) for n in cases}),indent=2),encoding='utf8')
    if not all(r['complete'] for rows in cases.values() for r in rows):raise SystemExit(1)


if __name__=='__main__':main()

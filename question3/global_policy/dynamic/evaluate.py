"""Paired local validation against saved fixed-ring cases, never official tests."""
import argparse
import json
from pathlib import Path
import numpy as np
from .runner import run_local
from .parameters import DynamicParameters


def evaluate_parameters(params,seeds):
    """Future optimizers call this local-only objective; no training is launched here."""
    seeds=list(seeds)
    if not seeds:raise ValueError('At least one evaluation seed is required')
    params.validate()
    rows=[run_local(seed,params)[0] for seed in seeds]
    failed=sum(not row['complete'] for row in rows)
    score=1e6+1e5*failed if failed else float(np.mean([row['average_time_s'] for row in rows]))
    return score,rows


def main():
    p=argparse.ArgumentParser();p.add_argument('--count',type=int,default=100)
    p.add_argument('--output',default='question3/global_policy/dynamic/outputs/paired_v2')
    p.add_argument('--params');args=p.parse_args()
    if not 1<=args.count<=100:p.error('--count must be between 1 and 100, matching the saved baseline')
    params=DynamicParameters(**json.loads(Path(args.params).read_text(encoding='utf8'))) if args.params else DynamicParameters()
    old=json.loads(Path('question3/global_policy/outputs/benchmark/cases.json').read_text(encoding='utf8'))['new'][:args.count]
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True);rows=[]
    decisions={};buckets={'20_to_60':{},'over_60':{}}
    for i,baseline in enumerate(old):
        seed=baseline['seed']
        row,planner=run_local(seed,params,out/f'seed_{seed}' if i<3 else None)
        rows.append(row)
        for event in planner.events:
            if event['phase']!='local_decision':continue
            reason=event['reason'];decisions[reason]=decisions.get(reason,0)+1
            bucket=buckets['20_to_60' if event['radius']<=60 else 'over_60'] if event['radius']>20 else None
            if bucket is not None:bucket[reason]=bucket.get(reason,0)+1
        if (i+1)%10==0 or not row['complete']:print(f'{i+1}/{len(old)} complete={row["complete"]} mean={row["average_time_s"]} failure={row["failure"]}',flush=True)
    (out/'cases.json').write_text(json.dumps(dict(dynamic=rows,fixed_ring=old),ensure_ascii=False,indent=2),encoding='utf8')
    if not all(row['complete'] for row in rows):
        failure_summary=dict(evaluation='local_synthetic_paired',count=len(rows),complete=sum(r['complete'] for r in rows),
                             failure='Incomplete runs; no successful-completion time comparison is claimed',
                             failed_seeds=[r['seed'] for r in rows if not r['complete']])
        (out/'summary.json').write_text(json.dumps(failure_summary,ensure_ascii=False,indent=2),encoding='utf8')
        raise SystemExit(1)
    paired=np.array([n['average_time_s']-b['average_time_s'] for n,b in zip(rows,old)])
    summary=dict(evaluation='local_synthetic_paired',count=len(rows),complete=sum(r['complete'] for r in rows),
        true_total=sum(r['true_total'] for r in rows),true_cleared=sum(r['true_cleared'] for r in rows),
        dynamic_mean_s=float(np.mean([r['average_time_s'] for r in rows])),fixed_ring_mean_s=float(np.mean([r['average_time_s'] for r in old])),
        paired_difference_s=float(paired.mean()),paired_se_s=float(paired.std(ddof=1)/np.sqrt(len(paired))) if len(paired)>1 else None,
        improved_cases=int((paired<0).sum()),mean_distance_m=float(np.mean([r['distance_m'] for r in rows])),
        fixed_ring_mean_distance_m=float(np.mean([r['distance_m'] for r in old])),
        mean_measures=float(np.mean([r['measures'] for r in rows])),misses=sum(r['misses'] for r in rows),
        mean_wall_time_s=float(np.mean([r['wall_time_s'] for r in rows])),local_choices=decisions,radius_buckets=buckets,
        frontier_departures_with_nearby=sum(r['frontier_departures_with_nearby'] for r in rows))
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    if not all(r['complete'] for r in rows):raise SystemExit(1)


if __name__=='__main__':main()

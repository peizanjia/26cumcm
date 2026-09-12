"""Offline clairvoyant bounds. This module never supplies data to a policy."""
import argparse,json
from pathlib import Path
import numpy as np
from question3.local_sim.simulator import random_scene
from .routes import held_karp


def scene_bounds(sources):
    points=np.array([[s.x,s.y] for s in sources]);n=len(points)
    start=np.linalg.norm(points,axis=1)
    edges=np.linalg.norm(points[:,None,:]-points[None,:,:],axis=2)
    zero=np.zeros(n)
    lower,_=held_karp(np.maximum(start-20,0),np.maximum(edges-40,0),zero)
    centers,order=held_karp(start,edges,zero)
    return dict(n=n,relaxed_distance_m=float(lower),oracle_center_distance_m=float(centers),
                general_lower_s_per_source=float(5+lower/(5*n)),
                mandatory_origin_scan_lower_s_per_source=float(5+(lower/5+119)/n),
                full_absent_coverage_lower_s_per_source=float(5+(lower/5+119+15*(20-n))/n),
                oracle_center_feasible_s_per_source=float(5+centers/(5*n)),
                center_order=[int(i) for i in order])


def main():
    p=argparse.ArgumentParser();p.add_argument('--count',type=int,default=100)
    p.add_argument('--seed-start',type=int,default=20263000)
    p.add_argument('--output',default='question3/global_policy/route_study/outputs/bounds.json');a=p.parse_args()
    rows=[dict(seed=seed,**scene_bounds(random_scene(seed))) for seed in range(a.seed_start,a.seed_start+a.count)]
    keys=['general_lower_s_per_source','mandatory_origin_scan_lower_s_per_source','full_absent_coverage_lower_s_per_source','oracle_center_feasible_s_per_source']
    summary={k:dict(mean=float(np.mean([r[k] for r in rows])),se=float(np.std([r[k] for r in rows],ddof=1)/np.sqrt(len(rows))),p10=float(np.percentile([r[k] for r in rows],10)),p90=float(np.percentile([r[k] for r in rows],90))) for k in keys}
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(dict(evaluation='offline_truth_bound_not_online_strategy',summary=summary,cases=rows),indent=2),encoding='utf8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()

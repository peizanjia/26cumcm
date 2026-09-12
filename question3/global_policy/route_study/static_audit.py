"""Compare algorithms on identical frozen public graphs, separate from online outcomes."""
import argparse,json
from pathlib import Path
import numpy as np
from ..model import World
from .parameters import StudyParameters
from .planner import run_local
from .routes import plan_route
from .graph_algorithms import length


def main():
    p=argparse.ArgumentParser();p.add_argument('--count',type=int,default=20);args=p.parse_args()
    methods=['insertion_2opt','nearest','exact_dp','mst_2opt','christofides','annealing']
    rows=[]
    for seed in range(20263000,20263000+args.count):
        _,planner=run_local(seed)
        for event in planner.events:
            if event['phase']!='route_problem' or len(event['channels'])<2:continue
            w=World(StudyParameters());w.position=np.array(event['position']);targets=[]
            for channel,center in zip(event['channels'],event['centers']):
                t=w.targets[channel];t.status='active';t.center=np.array(center);targets.append(t)
            end=np.array(event['destination']);points=np.array(event['centers']);channels=event['channels']
            costs={};orders={}
            for method in methods:
                order=plan_route(w,targets,end,method)
                orders[method]=[int(c) for c in order]
                costs[method]=length(points,w.position,end,[channels.index(c) for c in order])
            assert all(costs[m]>=costs['exact_dp']-1e-6 for m in methods)
            assert costs['annealing']<=costs['insertion_2opt']+1e-6
            rows.append(dict(seed=seed,size=len(channels),channels=channels,start=w.position.tolist(),centers=event['centers'],end=end.tolist(),costs_m=costs,orders=orders))
    summary={m:dict(mean_excess_over_exact_m=float(np.mean([r['costs_m'][m]-r['costs_m']['exact_dp'] for r in rows])),
                    exact_matches=sum(abs(r['costs_m'][m]-r['costs_m']['exact_dp'])<1e-6 for r in rows)) for m in methods}
    out=Path('question3/global_policy/route_study/outputs/static_audit.json')
    closed_surrogates=int(sum(np.linalg.norm(np.array(r['start'])-r['end'])<1e-7 for r in rows))
    out.write_text(json.dumps(dict(problem_count=len(rows),same_start_and_terminal=closed_surrogates,methods=summary,problems=rows),indent=2),encoding='utf8')
    print(json.dumps(dict(problem_count=len(rows),same_start_and_terminal=closed_surrogates,methods=summary),indent=2))


if __name__=='__main__':main()

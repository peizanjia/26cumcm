"""Paired local evaluation. Ground truth is used only after each policy finishes."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import numpy as np
from question3.local_sim.simulator import Simulator,Client
from question3.robot import Robot
from .runner import Planner,load_parameters


def run_new(seed,params,output=None,sources=None):
    sim=Simulator(seed,sources=sources)
    planner=Planner(Client(simulator=sim),params)
    summary=planner.run()
    n=len(sim.sources);removed=sum(s.cleared for s in sim.sources)
    summary.update(seed=seed,true_total=n,true_cleared=removed,clear_fraction=removed/n,
                   average_time_s=sim.virtual_time/removed if removed else None,evaluation='local_synthetic')
    summary['complete']=summary['complete'] and removed==n
    if output is not None:
        planner.save(output,summary);sim.dump(Path(output)/'evaluator')
    return summary


def run_old(seed):
    sim=Simulator(seed)
    robot=Robot('local')
    def post(path,payload):
        code,response=sim.handle(path,payload)
        if code!=200:raise RuntimeError((code,response))
        return response
    robot._post_once=post
    failure=None
    try:
        with contextlib.redirect_stdout(io.StringIO()):robot.run()
    except Exception as exc:failure=str(exc)
    n=len(sim.sources);removed=sum(s.cleared for s in sim.sources)
    return dict(seed=seed,true_total=n,true_cleared=removed,clear_fraction=removed/n,
                virtual_time_s=sim.virtual_time,average_time_s=sim.virtual_time/removed if removed else None,
                complete=removed==n and failure is None,failure=failure,
                measures=sum(h['path']=='/measure' for h in sim.history),
                clears=sum(h['path']=='/clear' for h in sim.history),
                misses=sum(h['resp'].get('clear_result')=='no_target_in_range' for h in sim.history))


def aggregate(rows):
    times=np.array([r['average_time_s'] for r in rows if r['average_time_s'] is not None])
    return dict(runs=len(rows),complete_runs=sum(r['complete'] for r in rows),
                sources=sum(r['true_total'] for r in rows),cleared=sum(r['true_cleared'] for r in rows),
                mean_average_time_s=float(times.mean()),median_average_time_s=float(np.median(times)),
                p90_average_time_s=float(np.quantile(times,.9)),
                se_average_time_s=float(times.std(ddof=1)/np.sqrt(len(times))) if len(times)>1 else None,
                mean_measures=float(np.mean([r['measures'] for r in rows])),
                mean_clears=float(np.mean([r['clears'] for r in rows])),
                misses=sum(r['misses'] for r in rows))


def evaluate_parameters(params,seeds):
    """Optimization entry: macro mean official per-case metric, fail-closed penalty."""
    rows=[run_new(int(seed),params) for seed in seeds]
    failures=sum(not r['complete'] for r in rows)
    return (1e6+1e5*failures if failures else float(np.mean([r['average_time_s'] for r in rows]))),rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',type=int,default=100)
    p.add_argument('--seed-start',type=int,default=20263000)
    p.add_argument('--params')
    p.add_argument('--skip-old',action='store_true')
    p.add_argument('--output',default='question3/global_policy/outputs/benchmark')
    args=p.parse_args()
    if args.runs<1:p.error('runs must be positive')
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    new=[];old=[];params=load_parameters(args.params)
    for i in range(args.runs):
        seed=args.seed_start+i
        new.append(run_new(seed,params,out/f'seed_{seed}' if i<3 else None))
        if not args.skip_old:old.append(run_old(seed))
        if (i+1)%10==0:print(f'{i+1}/{args.runs}: complete={sum(r["complete"] for r in new)}',flush=True)
    summary=dict(evaluation='local_synthetic',seed_start=args.seed_start,new=aggregate(new))
    if old:
        summary['old']=aggregate(old)
        good=[(a,b) for a,b in zip(new,old) if a['complete'] and b['complete']]
        if good:
            delta=np.array([a['average_time_s']-b['average_time_s'] for a,b in good])
            summary['paired_difference']=dict(n=len(good),mean_s=float(delta.mean()),
                se_s=float(delta.std(ddof=1)/np.sqrt(len(delta))) if len(delta)>1 else None,
                new_faster=int((delta<0).sum()))
    (out/'cases.json').write_text(json.dumps(dict(new=new,old=old),indent=2),encoding='utf8')
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    print(json.dumps(summary,indent=2))
    if not all(r['complete'] for r in new):raise SystemExit(1)


if __name__=='__main__':main()

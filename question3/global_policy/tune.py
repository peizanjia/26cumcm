"""Optional future parameter optimization; not run by default or by evaluation."""
import argparse
from dataclasses import asdict,replace
import json
from pathlib import Path
from scipy.optimize import differential_evolution
from .evaluate import evaluate_parameters,aggregate
from .model import Parameters

# Tune behavioral parameters, keeping physical constants and certification intact.
SEARCH_SPACE={
    'initial_step':(250.,800.),'initial_lateral':(0.,150.),
    'known_score_min':(.005,.08),'unknown_gain_min':(.03,.2),
    'service_radius':(30.,100.),'trial_probability_min':(.2,.8),
    'max_detour':(80.,400.),'ring_radius':(1250.,1550.)}


def parameters_from_vector(vector):
    return replace(Parameters(),**dict(zip(SEARCH_SPACE,map(float,vector)))).validate()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--iterations',type=int,default=5)
    parser.add_argument('--train-runs',type=int,default=8)
    parser.add_argument('--test-runs',type=int,default=30)
    parser.add_argument('--output',default='question3/global_policy/outputs/tuning')
    args=parser.parse_args()
    if min(args.iterations,args.train_runs,args.test_runs)<1:parser.error('Counts must be positive')
    if args.train_runs>1000:parser.error('Keep training seeds disjoint from held-out seeds')
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    train_seeds=range(20264000,20264000+args.train_runs)
    test_seeds=range(20265000,20265000+args.test_runs)
    history=[]
    def objective(vector):
        params=parameters_from_vector(vector)
        score,rows=evaluate_parameters(params,train_seeds)
        history.append(dict(parameters=asdict(params),score=score,complete_runs=sum(r['complete'] for r in rows)))
        print(f'evaluation={len(history)} score={score:.3f}',flush=True)
        return score
    base=Parameters();baseline_vector=[getattr(base,k) for k in SEARCH_SPACE]
    baseline_score=objective(baseline_vector)
    result=differential_evolution(objective,list(SEARCH_SPACE.values()),seed=20260912,
                                  maxiter=args.iterations,popsize=4,polish=False,x0=baseline_vector)
    best=parameters_from_vector(result.x) if result.fun<baseline_score else base
    _,held_out=evaluate_parameters(best,test_seeds)
    _,baseline_test=evaluate_parameters(base,test_seeds)
    (out/'best_parameters.json').write_text(json.dumps(asdict(best),indent=2),encoding='utf8')
    (out/'optimization.json').write_text(json.dumps(dict(success=bool(result.success),message=str(result.message),
        nfev=int(result.nfev),history=history,held_out=aggregate(held_out),baseline_held_out=aggregate(baseline_test)),indent=2),encoding='utf8')
    # A budget-limited DE result is a tested candidate, not a global optimum.
    print(json.dumps(dict(held_out=aggregate(held_out),baseline_held_out=aggregate(baseline_test)),indent=2))


if __name__=='__main__':main()

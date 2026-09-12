"""Run a selected policy locally and save its effective configuration."""
import argparse,json
from dataclasses import asdict
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description='Local synthetic Q3 only; no official connection')
    p.add_argument('--seed',type=int,default=20272000)
    p.add_argument('--engine', choices=['tour','sweep','global','combined'], default='tour',
                   help='tour: joint source/coverage route; global: experimental action-value surrogate')
    p.add_argument('--params');p.add_argument('--output',default='question3/global_policy/adaptive_mpc/outputs/demo')
    a=p.parse_args();fields=json.loads(Path(a.params).read_text(encoding='utf8')) if a.params else {}
    if a.engine in ('tour','sweep'):
        from .sweep_planner import SweepParameters
        from .benchmark import VARIANTS
        defaults={k:v for k,v in VARIANTS[a.engine+'_candidate'].items() if k!='engine'}
        defaults.update(fields)
        parameters=SweepParameters(**defaults)
        if a.engine=='tour':from .tour_planner import run_local
        else:from .sweep_planner import run_local
    elif a.engine=='global':
        from .planner import run_local
        from .parameters import AdaptiveParameters
        parameters=AdaptiveParameters(**fields)
    else:
        from ..joint_rollout.planner import run_local
        from ..joint_rollout.parameters import JointParameters
        parameters=JointParameters(**fields)
    parameters.validate()
    output=Path(a.output)
    if (output/'selected_configuration.json').exists():
        p.error('Output already contains a run; choose a fresh --output directory')
    output.mkdir(parents=True,exist_ok=True)
    (output/'selected_configuration.json').write_text(json.dumps(dict(
        engine=a.engine,seed=a.seed,parameters=asdict(parameters),
        evaluation='local_synthetic'),ensure_ascii=False,indent=2),encoding='utf8')
    row,_=run_local(a.seed,parameters,a.output)
    print(json.dumps(row,ensure_ascii=False,indent=2))
    if not row['complete']:raise SystemExit(1)

if __name__=='__main__':main()

"""Run one experimental scenario, locally only."""
import argparse,json
from pathlib import Path
from .parameters import StudyParameters
from .planner import run_local
from .benchmark import VARIANTS


def main():
    p=argparse.ArgumentParser();p.add_argument('--variant',choices=list(VARIANTS),default='baseline')
    p.add_argument('--seed',type=int,default=20260911);p.add_argument('--params')
    p.add_argument('--output',default='question3/global_policy/route_study/outputs/demo');a=p.parse_args()
    fields=dict(VARIANTS[a.variant])
    if a.params:fields.update(json.loads(Path(a.params).read_text(encoding='utf8')))
    summary,_=run_local(a.seed,StudyParameters(**fields),a.output)
    print(json.dumps(summary,indent=2))
    if not summary['complete']:raise SystemExit(1)


if __name__=='__main__':main()

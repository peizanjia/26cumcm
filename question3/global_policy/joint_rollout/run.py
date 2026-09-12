import argparse,json
from .parameters import JointParameters
from .planner import run_local


VARIANTS={
    'baseline':dict(linked_terminal=False,continuous_service=False,samples=256),
    'terminal':dict(linked_terminal=True,continuous_service=False,samples=256),
    'expectation':dict(linked_terminal=False,continuous_service=True),
    'combined':dict(linked_terminal=True,continuous_service=True),
}


def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=20260911)
    p.add_argument('--variant',choices=list(VARIANTS),default='combined');p.add_argument('--params')
    p.add_argument('--output',default='question3/global_policy/joint_rollout/outputs/demo');a=p.parse_args()
    fields=dict(VARIANTS[a.variant])
    if a.params:fields.update(json.loads(open(a.params,encoding='utf8').read()))
    row,_=run_local(a.seed,JointParameters(**fields),a.output)
    print(json.dumps(row,indent=2))
    if not row['complete']:raise SystemExit(1)


if __name__=='__main__':main()

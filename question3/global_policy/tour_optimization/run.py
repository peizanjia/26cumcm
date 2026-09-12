"""Reproduce the selected policy locally; effective configuration is recorded."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[key]='1'
import argparse,json
from pathlib import Path
from dataclasses import asdict
from .parameters import Parameters
from .planner import run_local


def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=20273700)
    p.add_argument('--params',default=str(Path(__file__).with_name('best_parameters.json')))
    p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output)
    if out.exists() and any(out.iterdir()):p.error('Use a new output directory')
    params=Parameters(**json.loads(Path(a.params).read_text(encoding='utf8'))).validate()
    out.mkdir(parents=True,exist_ok=True)
    (out/'selected_configuration.json').write_text(json.dumps(dict(seed=a.seed,evaluation='local_synthetic',
        parameters=asdict(params)),indent=2),encoding='utf8')
    row,_=run_local(a.seed,params,out)
    print(json.dumps(row,indent=2),flush=True)
    if not row['complete']:raise SystemExit(1)


if __name__=='__main__':main()

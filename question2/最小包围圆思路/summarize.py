"""Rebuild descriptive statistics from saved held-out outcomes, without training."""
import csv
import json
from pathlib import Path
import numpy as np
from .core import statistics,dataset

ROOT=Path(__file__).parent


def main():
    record=json.loads((ROOT/'evaluation.json').read_text())
    raw=np.load(ROOT/'evaluation_samples.npz')
    d=dataset(1024,256,29260911)
    for name,v in record['policies'].items():
        q=raw[name+'_q'];r=raw[name+'_r']
        v['movement_m']=statistics(np.linalg.norm(q.astype(float),axis=1),threshold=False)
        distance=np.linalg.norm(d['targets']-q[:,None],axis=-1)
        v['no_signal_count']=int((distance>d['radii']).sum())
        v['near_count']=int((distance<=5).sum())
    (ROOT/'evaluation.json').write_text(json.dumps(record,indent=2))
    rows=[dict(policy=n,metric=m,**st) for n,v in record['policies'].items() for m,st in v.items() if isinstance(st,dict)]
    with (ROOT/'statistics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print({k:dict(mean=v['radius_m']['mean'],no_signal=v['no_signal_count'],near=v['near_count']) for k,v in record['policies'].items()})


if __name__=='__main__':main()

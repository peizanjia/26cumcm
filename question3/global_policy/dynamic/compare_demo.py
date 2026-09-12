"""Audit the user's previous demo against the replacement on the same seed."""
import json
from pathlib import Path
import numpy as np


def metrics(folder):
    folder=Path(folder);summary=json.loads((folder/'summary.json').read_text(encoding='utf8'))
    commands=[json.loads(x) for x in (folder/'commands.jsonl').read_text(encoding='utf8').splitlines()]
    points=[np.zeros(2)]
    for command in commands:
        q=command['position']
        if q is not None and np.linalg.norm(np.asarray(q)-points[-1])>1e-6:points.append(np.asarray(q))
    vectors=np.diff(points,axis=0);length=np.linalg.norm(vectors,axis=1)
    cosine=np.sum(vectors[:-1]*vectors[1:],axis=1)/(length[:-1]*length[1:])
    reversal=(length[:-1]>600)&(length[1:]>600)&(cosine<-.5)
    return dict(seed=summary['seed'],average_time_s=summary['average_time_s'],distance_m=summary['distance_m'],
                long_reversals=int(reversal.sum()),max_segment_m=float(length.max()),misses=summary['misses'])


def main():
    result=dict(definition='Two consecutive nonzero movement segments each >600m, turning angle >120 degrees. Diagnostic only, not a universal route bound.',
                fixed_ring=metrics('question3/global_policy/outputs/final_demo'),
                dynamic=metrics('question3/global_policy/dynamic/outputs/demo'))
    path=Path('question3/global_policy/dynamic/outputs/demo/comparison.json')
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()

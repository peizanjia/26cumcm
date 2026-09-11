"""Paired target-20 / orthogonality fine-tuning, no optimal-action labels."""
import argparse
import csv
import json
import time
from pathlib import Path
import numpy as np
import torch
from question2.最小包围圆思路.core import dataset
from ..symmetry.policy import TwoSidedNet
from .core import expected_components

ROOT=Path(__file__).parent


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--steps',type=int,default=1800)
    args=parser.parse_args()
    torch.set_num_threads(4)
    ck=ROOT/'checkpoints';ck.mkdir(exist_ok=True)
    results=ROOT/'results';results.mkdir(exist_ok=True)
    rng=np.random.default_rng(33260911)
    data=dataset(4096,128,33260911);val=dataset(384,192,34260911)
    origin=ROOT.parents[1]/'最小包围圆思路/radius_nn.pt'
    payload=torch.load(origin,weights_only=True,map_location='cpu')
    configs=[(rep,w) for rep in range(2) for w in (0.,.1,.5)]
    nets=[]
    for rep,w in configs:
        net=TwoSidedNet(boundary_scaled=True);net.load_state_dict(payload['state_dict'])
        torch.manual_seed(36260911+rep)
        with torch.no_grad():
            for par in net.parameters():par.add_(torch.randn_like(par)*(.0005 if rep else 0.))
        nets.append(net)
    optim=torch.optim.Adam([p for net in nets for p in net.parameters()],lr=2e-4)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(optim,args.steps,eta_min=1e-5)
    weights=np.array([w for rep,w in configs])
    best=np.full(len(nets),np.inf);best_rows={};history=[];start=time.perf_counter()
    for step in range(args.steps+1):
        if step%300==0 or step==args.steps:
            for k,net in enumerate(nets):
                with torch.no_grad():qs=net(torch.tensor(val['states'][:,3:5],dtype=torch.float32)).numpy()*1500
                cs=np.stack([expected_components(val,qs[:,side]) for side in (0,1)])
                avg=cs.mean((0,1));objective=float(avg[0]+weights[k]*avg[1])
                row=dict(step=step,replicate=configs[k][0],orth_weight=float(weights[k]),
                    objective=objective,mae20_m=float(avg[0]*20),orth=float(avg[1]),
                    mean_radius_m=float(avg[2]),no_signal=float(avg[3]))
                history.append(row)
                if objective<best[k]:
                    best[k]=objective;best_rows[k]=row
                    torch.save(dict(state_dict=net.state_dict(),boundary_scaled=True,
                        **row,training_seed=33260911,validation_seed=34260911),ck/f'model_{k}.pt')
            with (results/'training.csv').open('w',newline='') as f:
                wr=csv.DictWriter(f,fieldnames=list(history[0]));wr.writeheader();wr.writerows(history)
            print(json.dumps(dict(step=step,elapsed_s=time.perf_counter()-start,validation=history[-6:])),flush=True)
        if step==args.steps:break
        ix=rng.integers(0,len(data['states']),64);js=rng.integers(0,128,32)
        d={k:v[ix] for k,v in data.items()}
        for key in ('targets','errors','radii'):d[key]=d[key][:,js]
        p=torch.tensor(d['states'][:,3:5],dtype=torch.float32)
        actions=[net(p)[:,step%2]*1500 for net in nets]
        q=torch.cat(actions);qn=q.detach().numpy().astype(float)
        dd={k:np.tile(v,(len(nets),)+(1,)*(v.ndim-1)) for k,v in d.items()}
        ww=np.repeat(weights,len(ix));grad=np.empty_like(qn)
        for axis in (0,1):
            shift=np.zeros_like(qn);shift[:,axis]=.5
            plus=expected_components(dd,qn+shift);minus=expected_components(dd,qn-shift)
            grad[:,axis]=(plus[:,0]-minus[:,0])+ww*(plus[:,1]-minus[:,1])
        if not np.isfinite(grad).all():raise FloatingPointError('Nonfinite action gradient')
        optim.zero_grad(set_to_none=True)
        # Divide by states, not models: each independently parameterized model has its own mean loss.
        surrogate=(q*torch.tensor(grad,dtype=q.dtype)).sum()/len(ix)
        surrogate.backward()
        for net in nets:torch.nn.utils.clip_grad_norm_(net.parameters(),5.)
        optim.step();sched.step()
    selected={}
    for name,indices in [('plain',[0,3]),('orthogonal',[1,2,4,5])]:
        # Common primary validation metric selects the deployable model, never test data.
        k=min(indices,key=lambda j:best_rows[j]['mae20_m'])
        torch.save(torch.load(ck/f'model_{k}.pt',weights_only=True),ck/f'{name}.pt')
        selected[name]=dict(model=k,**best_rows[k])
    (results/'training_summary.json').write_text(json.dumps(dict(
        steps=args.steps,states=4096,events=128,batch_states=64,batch_events=32,
        seed=33260911,validation_seed=34260911,replicate_initialization_seeds=[36260911,36260912],
        warm_start=str(origin),finite_difference_h_m=.5,configs=configs,selected=selected,
        elapsed_s=time.perf_counter()-start),indent=2))


if __name__=='__main__':main()

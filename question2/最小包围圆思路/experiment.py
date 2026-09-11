"""Radius-only and threshold/distance-guided policies; finite-difference action gradients."""
import csv
import json
import time
from pathlib import Path
import numpy as np
import torch
from .core import dataset,evaluate,statistics,expected_cost
from question2.黑箱思路.symmetry.policy import TwoSidedNet
from question2.黑箱思路.analytic_baseline import moment_action

ROOT=Path(__file__).parent


def cost(r,q,guided):
    v=r.mean(1)/20
    if guided: v=v+4*np.mean(np.maximum(r/20-1,0)**2,axis=1)+.15*np.linalg.norm(q,axis=1)/1500
    return v


def main():
    torch.set_num_threads(4)
    rng=np.random.default_rng(27260911)
    d=dataset(2048,96,27260911)
    val=dataset(256,128,28260911)
    oldpath=ROOT.parent/'黑箱思路/checkpoints/symmetry/best.pt'
    old=torch.load(oldpath,weights_only=True,map_location='cpu')
    nets=[TwoSidedNet(boundary_scaled=True) for _ in range(2)]
    for net in nets: net.load_state_dict(old['state_dict'])
    optim=[torch.optim.Adam(net.parameters(),lr=3e-4) for net in nets]
    names=['radius_nn','guided_nn'];best=[np.inf,np.inf];history=[]
    start=time.perf_counter()
    for step in range(801):
        if step%100==0:
            for k,net in enumerate(nets):
                with torch.no_grad(): q=net(torch.tensor(val['states'][:,3:5],dtype=torch.float32))[:,0].numpy()*1500
                r=evaluate(val,q);c=float(expected_cost(val,q,k==1).mean())
                history.append(dict(step=step,policy=names[k],loss=c,mean=float(r.mean()),success=float((r<=20).mean())))
                if c<best[k]:
                    best[k]=c;torch.save(dict(state_dict=net.state_dict(),boundary_scaled=True,step=step,loss=c),ROOT/(names[k]+'.pt'))
            print(json.dumps(dict(step=step,elapsed=time.perf_counter()-start,validation=history[-2:])),flush=True)
        if step==800: break
        ix=rng.integers(0,2048,48);js=rng.integers(0,96,24)
        dd={key:value[ix] for key,value in d.items()}
        for key in ('targets','errors','radii'):dd[key]=dd[key][:,js]
        p=torch.tensor(dd['states'][:,3:5],dtype=torch.float32)
        # Train both mirrored branches without hidden-target side selection.
        side=step%2
        for k,net in enumerate(nets):
            q=net(p)[:,side]*1500;qn=q.detach().numpy().astype(float)
            grad=np.empty_like(qn)
            for axis in (0,1):
                shift=np.zeros_like(qn);shift[:,axis]=.5
                plus=qn+shift;minus=qn-shift
                grad[:,axis]=(expected_cost(dd,plus,k==1)-expected_cost(dd,minus,k==1))
            optim[k].zero_grad()
            (q*torch.tensor(grad,dtype=q.dtype)).sum(1).mean().backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(),5.)
            optim[k].step()
    with (ROOT/'training.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(history[0]));w.writeheader();w.writerows(history)
    test=dataset(1024,256,29260911)
    oldnet=TwoSidedNet(boundary_scaled=True);oldnet.load_state_dict(old['state_dict'])
    p=torch.tensor(test['states'][:,3:5],dtype=torch.float32)
    with torch.no_grad():oldq=oldnet(p)[:,0].numpy()*1500
    actions={'old_area_nn':oldq,'area_moment_formula':moment_action(test['states'][:,3:5])*1500}
    for name,net in zip(names,nets):
        net.load_state_dict(torch.load(ROOT/(name+'.pt'),weights_only=True)['state_dict'])
        with torch.no_grad():actions[name]=net(p)[:,0].numpy()*1500
    records={};raw={}
    for name,q in actions.items():
        r=evaluate(test,q);records[name]=dict(radius_m=statistics(r),movement_m=statistics(np.linalg.norm(q,axis=1),threshold=False),
            state_expected_radius_m=statistics(r.mean(1)),fraction_states_samplemax_le20=float(np.mean(r.max(1)<=20)))
        raw[name+'_r']=r;raw[name+'_q']=q
    np.savez_compressed(ROOT/'evaluation_samples.npz',**raw)
    (ROOT/'evaluation.json').write_text(json.dumps(dict(test_seed=29260911,states=1024,events_per_state=256,
        variance_definition='population variance ddof=0; pooled events are clustered by state',
        elapsed_s=time.perf_counter()-start,policies=records),indent=2))
    with (ROOT/'statistics.csv').open('w',newline='') as f:
        rows=[dict(policy=n,metric=m,**st) for n,v in records.items() for m,st in v.items() if isinstance(st,dict)]
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps(records),flush=True)


if __name__=='__main__':main()

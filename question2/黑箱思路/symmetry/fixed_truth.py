"""Oracle diagnostic landscapes: fixed first observation AND hidden target.

Only epsilon_2 is integrated by deterministic midpoint quadrature. Hidden R
is fixed at 1250 m for these diagnostics. Policies never receive that truth.
"""
import csv
import json
from pathlib import Path
import numpy as np
import torch
from ..geometry import SCALE,DELTA,first_polygon,area_cpu
from ..objective import outcomes,tensor_dataset
from ..policy import NeuralPolicy
from ..analytic_baseline import moment_action
from .policy import SidePolicy

ROOT=Path(__file__).parents[1]
CASES=[dict(tag="near400",s=[0,0],g=[400,0],theta=0),
       dict(tag="mid800",s=[0,0],g=[800,0],theta=0),
       dict(tag="far1200",s=[0,0],g=[1200,0],theta=0),
       dict(tag="edge_out",s=[1500,0],g=[1750,0],theta=0),
       dict(tag="oblique_boundary",s=[1000,1200],g=[1250,1200],theta=0),
       dict(tag="first_error",s=[0,0],g=[1000*np.cos(np.deg2rad(.8)),-1000*np.sin(np.deg2rad(.8))],theta=0)]


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    device="cuda" if torch.cuda.is_available() else "cpu"
    output=ROOT/"results"/"symmetry"/"fixed_truth";output.mkdir(exist_ok=True)
    old=NeuralPolicy(device=device); new=SidePolicy(device=device)
    fixed=json.loads((output.parent/"optimized_fixed.json").read_text())
    metadata=[]
    for case in CASES:
        s=np.array(case["s"],float);g=np.array(case["g"],float)
        t=np.deg2rad(case["theta"]);rot=np.array([[np.cos(t),-np.sin(t)],[np.sin(t),np.cos(t)]])
        p=np.einsum("ji,j->i",rot,s)/SCALE
        target=np.einsum("ji,j->i",rot,g-s)/SCALE
        poly=first_polygon(p,1024)
        n=256
        errors=((np.arange(n)+.5)/n*2-1)*DELTA
        d=tensor_dataset(dict(states=np.array([[*(s/SCALE),t,*p]]),polygons=poly[None],counts=np.array([len(poly)]),
                              areas=np.array([area_cpu(poly)]),targets=np.broadcast_to(target,(1,n,2)).copy(),errors=errors[None]),device)
        xx,yy=np.meshgrid(np.linspace(-200,1800,201),np.linspace(-900,900,181))
        if case["tag"] in ("edge_out","oblique_boundary"):
            xx,yy=np.meshgrid(np.linspace(-100,600,201),np.linspace(-300,300,181))
        q=np.column_stack([xx.ravel(),yy.ravel()])/SCALE
        scores=[];nosignal=[]
        for start in range(0,len(q),128):
            qq=torch.tensor(q[start:start+128],device=device,dtype=torch.float32)
            dd={k:v.expand((len(qq),)+v.shape[1:]) for k,v in d.items()}
            r=outcomes(dd,qq,radius_mode="1250")
            scores.extend(r["loss"].mean(1).cpu().numpy());nosignal.extend(r["no_signal"].mean(1).cpu().numpy())
        scores=np.array(scores);nosignal=np.array(nosignal)
        oldq=old.net(torch.tensor(p,device=device,dtype=torch.float32)).cpu().numpy()*SCALE
        newq=new.net(torch.tensor(p,device=device,dtype=torch.float32)).cpu().numpy()*SCALE
        aq=moment_action(p)[0]*SCALE
        case.update(radius_m=1250,second_error_midpoints=n,
                    first_error_deg=float(case["theta"]-np.degrees(np.arctan2((g-s)[1],(g-s)[0]))),
                    score_definition="E_epsilon2 conservative residual area; S,G,theta1,R held fixed; near after optical=0",
                    caveat="Diagnostic oracle truth is NOT available to any policy")
        metadata.append(case)
        np.savez_compressed(output/(case["tag"]+".npz"),xx=xx,yy=yy,score=scores.reshape(xx.shape),
                            no_signal=nosignal.reshape(xx.shape),target=target*SCALE,polygon=poly*SCALE,
                            old=oldq,new=newq,approx=aq,fixed=[fixed["a_m"],-fixed["abs_b_m"]])
        with (output/(case["tag"]+".csv")).open("w",newline="") as f:
            w=csv.writer(f);w.writerow(["along_m","lateral_m","expected_area_m2","no_signal"])
            w.writerows(zip(xx.ravel(),yy.ravel(),scores,nosignal))
        print(case["tag"],float(scores.min()),flush=True)
    (output/"cases.json").write_text(json.dumps(metadata,indent=2))


if __name__=="__main__":main()

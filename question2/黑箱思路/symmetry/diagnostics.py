"""Boundary scans, approximation audit, and a constant-policy loss landscape."""
import json
from pathlib import Path
import numpy as np
import torch
from ..candidates import state_data,score_grid
from ..circle_reference import exact_circle_area
from ..environment import make_dataset
from ..objective import tensor_dataset,outcomes
from ..geometry import SCALE,DELTA
from ..analytic_baseline import moment_action
from ..policy import NeuralPolicy
from .policy import SidePolicy

ROOT=Path(__file__).parents[1]


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    device="cuda" if torch.cuda.is_available() else "cpu"
    output=ROOT/"results"/"symmetry"
    old=NeuralPolicy(device=device).net;new=SidePolicy(device=device).net
    lengths=np.linspace(30,1500,100)
    p=np.column_stack([(1800-lengths)/1500,np.zeros(len(lengths))])
    pp=torch.tensor(p,device=device,dtype=torch.float32)
    np.savez_compressed(output/"boundary_curve.npz",length_m=lengths,
                        old=old(pp).cpu().numpy()*1500,new=new(pp)[:,0].cpu().numpy()*1500,
                        approximate=moment_action(p)*1500)
    scans=[]
    for i,length in enumerate([80,150,300,600,1000,1250,1500]):
        data,local_p,poly=state_data(1800-length,0,0,8192,5630+i)
        lp=torch.tensor(local_p,device=device,dtype=torch.float32)
        actions=np.stack([old(lp).cpu().numpy(),new(lp)[0].cpu().numpy(),moment_action(local_p)[0]])
        scores,failures,se=score_grid(data,actions,device)
        a,b=actions[2]
        radii=np.linalg.norm(data["targets"][0],axis=1)
        approx=4*DELTA**2*radii*((radii-a)**2+b*b)/abs(b)*SCALE**2
        # Independently integrate the REAL circle for 64 of these SAME events.
        td=tensor_dataset({k:(v[:,:64] if k in ("targets","errors") else v) for k,v in data.items()},device)
        measured=outcomes(td,torch.tensor(actions[2:3],device=device,dtype=torch.float32))["area"].cpu().numpy()[0]
        exact=[]
        for j,g in enumerate(data["targets"][0,:64]):
            t=np.arctan2(*(g-actions[2])[::-1])+data["errors"][0,j]
            exact.append(exact_circle_area(local_p,actions[2],t)*SCALE**2)
        scans.append(dict(length_m=length,actions_m=(actions*SCALE).tolist(),
                          original_nn_m2=float(scores[0]),shared_nn_m2=float(scores[1]),
                          moments_true_objective_m2=float(scores[2]),mc_se_m2=se.tolist(),
                          moments_approximate_objective_m2=float(approx.mean()),
                          circle_polygon_mean_abs_error_m2=float(np.mean(abs(measured-np.array(exact)))),
                          circle_polygon_max_abs_error_m2=float(np.max(abs(measured-np.array(exact))))))
        print(json.dumps(scans[-1]),flush=True)
    # One score landscape on an optimization split, not the held-out test set.
    d=tensor_dataset(make_dataset(512,64,21460911),device)
    xx,yy=np.meshgrid(np.linspace(650,1150,41),np.linspace(150,450,41))
    actions=np.column_stack([xx.ravel(),-yy.ravel()])/1500
    costs=[]
    for start in range(0,len(actions),8):
        points=torch.tensor(actions[start:start+8],device=device,dtype=torch.float32)
        dd={k:v.repeat((len(points),)+(1,)*(v.ndim-1)) for k,v in d.items()}
        qq=points[:,None].expand(-1,len(d["states"]),-1).reshape(-1,2)
        costs.extend(outcomes(dd,qq)["loss"].reshape(len(points),-1).mean(1).cpu().numpy())
    np.savez_compressed(output/"fixed_landscape.npz",xx=xx,yy=yy,score=np.array(costs).reshape(xx.shape))
    (output/"boundary_audit.json").write_text(json.dumps(scans,indent=2))


if __name__=="__main__":main()

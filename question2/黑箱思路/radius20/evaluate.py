"""Independent paired test, event distribution and state-cluster intervals."""
import csv
import json
from pathlib import Path
import numpy as np
import torch
from question2.最小包围圆思路.core import dataset,evaluate,statistics
from ..symmetry.policy import TwoSidedNet
from .core import expected_components

ROOT=Path(__file__).parent


def summary(r):
    st=statistics(r);x=np.asarray(r).ravel()
    st.update(mae20=float(np.abs(x-20).mean()),rmse20=float(np.sqrt(np.mean((x-20)**2))),
              p01=float(np.quantile(x,.01)),p10=float(np.quantile(x,.1)),
              iqr=float(np.quantile(x,.75)-np.quantile(x,.25)),
              p_le50=float(np.mean(x<=50)),p_within2_of20=float(np.mean(np.abs(x-20)<=2)),
              p_within5_of20=float(np.mean(np.abs(x-20)<=5)))
    return st


def main():
    torch.set_num_threads(4)
    d=dataset(2048,256,35260911)
    p=torch.tensor(d['states'][:,3:5],dtype=torch.float32)
    paths={'plain':ROOT/'checkpoints/plain.pt','orthogonal':ROOT/'checkpoints/orthogonal.pt',
           'previous_radius_min':ROOT.parents[1]/'最小包围圆思路/radius_nn.pt',
           'previous_threshold_distance':ROOT.parents[1]/'最小包围圆思路/guided_nn.pt'}
    rows={};raw={};configs={}
    for name,path in paths.items():
        cp=torch.load(path,weights_only=True,map_location='cpu')
        net=TwoSidedNet(boundary_scaled=True);net.load_state_dict(cp['state_dict'])
        with torch.no_grad():q=net(p).numpy()*1500
        radii=[];angles=[];directions=[];no=[];near=[];components=[]
        for side in (0,1):
            qq=q[:,side];r=evaluate(d,qq);radii.append(r)
            diff=d['targets']-qq[:,None];dist=np.linalg.norm(diff,axis=-1)
            same=np.linalg.norm(qq,axis=-1)<1e-9
            ns=(dist>d['radii']) & ~same[:,None];nr=dist<=5
            good=~ns & ~nr & ~same[:,None]
            theta=np.arctan2(diff[...,1],diff[...,0])+d['errors']
            deviation=np.rad2deg(np.arcsin(np.clip(np.abs(np.cos(theta)),0,1)))
            angles.append(deviation);directions.append(good);no.append(ns);near.append(nr)
            components.append(expected_components(d,qq))
        # state remains the first axis; sides are equally weighted, no oracle side choice.
        r=np.stack(radii,1);a=np.stack(angles,1);good=np.stack(directions,1)
        c=np.stack(components,1)
        state_mae=np.abs(r-20).mean((1,2))
        rows[name]=dict(radius_m=summary(r),state_expected_radius_m=summary(r.mean((1,2))),
            movement_m=statistics(np.linalg.norm(q,axis=-1),threshold=False),
            mean_abs20_marginalized=float(c[...,0].mean()*20),
            available_weighted_cos2=float(c[...,1].mean()),
            angle_deviation_from90_deg=statistics(a[good],threshold=False),
            p_direction_within10deg_of90=float((a[good]<=10).mean()),
            no_signal_rate=float(np.mean(no)),near_rate=float(np.mean(near)),
            fraction_state_side_samplemax_le20=float(np.mean(r.max(2)<=20)),
            mae20_cluster95=[float(state_mae.mean()-1.96*state_mae.std(ddof=1)/np.sqrt(len(r))),
                             float(state_mae.mean()+1.96*state_mae.std(ddof=1)/np.sqrt(len(r)))])
        raw[name+'_r']=r;raw[name+'_q']=q;raw[name+'_angle']=a;raw[name+'_direction']=good
        configs[name]={k:v for k,v in cp.items() if k!='state_dict'}
        print(name,json.dumps(rows[name]['radius_m']),flush=True)
    delta=(np.abs(raw['orthogonal_r']-20)-np.abs(raw['plain_r']-20)).mean((1,2))
    paired=dict(metric='guided minus plain MAE20 in metres, lower is better',mean=float(delta.mean()),
        cluster_normal95=[float(delta.mean()-1.96*delta.std(ddof=1)/np.sqrt(len(delta))),
                          float(delta.mean()+1.96*delta.std(ddof=1)/np.sqrt(len(delta)))])
    output=ROOT/'results'
    np.savez_compressed(output/'evaluation_samples.npz',**raw)
    (output/'evaluation.json').write_text(json.dumps(dict(seed=35260911,states=2048,events_per_state=256,
        sides='equally weighted; not independent scenarios',variance='population ddof=0',
        policies=rows,paired_mae20=paired,checkpoints=configs),indent=2))
    records=[dict(policy=k,**v['radius_m'],mean_angle_error90=v['angle_deviation_from90_deg']['mean'],
                  no_signal=v['no_signal_rate'],mean_movement=v['movement_m']['mean']) for k,v in rows.items()]
    with (output/'statistics.csv').open('w',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=list(records[0]));wr.writeheader();wr.writerows(records)


if __name__=='__main__':main()

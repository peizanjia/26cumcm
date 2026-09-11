"""New held-out test; preserve the first study and all its original results."""
import csv
import json
from pathlib import Path
import numpy as np
import torch
from ..environment import make_dataset
from ..objective import tensor_dataset, evaluate
from ..policy import NeuralPolicy
from ..analytic_baseline import moment_action
from ..evaluate import summarize
from .policy import SidePolicy, ray_length

ROOT = Path(__file__).parents[1]


def stats(values):
    values=np.asarray(values, float)
    return dict(mean=float(values.mean()), std=float(values.std()),
                p05=float(np.quantile(values,.05)), p50=float(np.median(values)), p95=float(np.quantile(values,.95)))


def main():
    torch.set_num_threads(4)
    device="cuda" if torch.cuda.is_available() else "cpu"
    output=ROOT/"results"/"symmetry"
    data=make_dataset(8192, 256, 23260911)
    td=tensor_dataset(data,device)
    old=NeuralPolicy(device=device).net
    new=SidePolicy(device=device).net
    fixed=json.loads((output/"optimized_fixed.json").read_text())
    def constant(a,b):
        return lambda d: d["states"].new_tensor([a/1500,b/1500]).expand(len(d["states"]),-1)
    methods={"original_nn":lambda d:old(d["states"][:,3:5]),
             "shared_right":lambda d:new(d["states"][:,3:5])[:,0],
             "shared_left":lambda d:new(d["states"][:,3:5])[:,1],
             "hand_fixed_750_375":constant(750,-375),
             "optimized_fixed":constant(fixed["a_m"],-fixed["abs_b_m"]),
             "approximate_moments":lambda d:torch.tensor(moment_action(d["states"][:,3:5].cpu().numpy()),device=device,dtype=torch.float32)}
    results={}; rows=[]
    for name,action in methods.items():
        results[name]=evaluate(td,action)
        row=summarize(name,results[name],data); rows.append(row)
        print(json.dumps(row),flush=True)
    # Fair external-side comparison is the equal-weight side mean, NOT choosing
    # the side with the smaller realized true-target loss.
    right=results["shared_right"]["loss"].mean(1)
    left=results["shared_left"]["loss"].mean(1)
    side_average=(right+left)/2
    paired={}
    for name,r in results.items():
        delta=r["loss"].mean(1)-side_average
        se=delta.std(ddof=1)/np.sqrt(len(delta))
        paired[name]=dict(baseline_minus_shared_average_m2=float(delta.mean()),
                         ci95=[float(delta.mean()-1.96*se),float(delta.mean()+1.96*se)])
    p=data["states"][:,3:5]
    # Fully unaffected by the arena: the complete 1500 m / ±1° sector is inside.
    alpha=np.linspace(-np.pi/180,np.pi/180,65)
    arc=np.stack([np.cos(alpha),np.sin(alpha)],-1)
    full=np.max(np.linalg.norm(p[:,None]+arc[None],axis=2),axis=1)<=1.2
    length=ray_length(td["states"][:,3:5]).cpu().numpy()*1500
    relation={}
    for name in ("original_nn","shared_right","approximate_moments"):
        q=results[name]["q"]*1500
        relation[name]={"all_forward_m":stats(q[:,0]),"all_abs_lateral_m":stats(abs(q[:,1])),
                        "full_sector_forward_m":stats(q[full,0]),"full_sector_abs_lateral_m":stats(abs(q[full,1]))}
    delta=(results["original_nn"]["q"]-results["approximate_moments"]["q"])*1500
    relation["original_vs_moments"]={"rms_displacement_difference_m":float(np.sqrt(np.mean((delta**2).sum(1)))),
                                   "full_sector_rms_difference_m":float(np.sqrt(np.mean((delta[full]**2).sum(1))))}
    with torch.inference_mode():
        pp=td["states"][:,3:5]
        f=pp.new_tensor([1.,-1.])
        error=(new(pp*f)-new(pp).flip(-2)*f).abs().max().item()*1500
    report=dict(test_seed=23260911,states=8192,samples=256,summary=rows,
                external_fair_coin_mean_m2=float(side_average.mean()),paired=paired,
                optimized_fixed=fixed,full_sector_state_count=int(full.sum()),relations=relation,
                architecture_reflection_max_error_m=error,
                same_state_side_score_difference_m2=stats(abs(left-right)),
                caveat="Both sides evaluated separately; no oracle per-target side choice")
    (output/"evaluation.json").write_text(json.dumps(report,indent=2))
    np.savez_compressed(output/"state_actions.npz",states=data["states"],length_m=length,full_sector=full,
                        **{name+"_q":r["q"]*1500 for name,r in results.items()},
                        right_mean_m2=right,left_mean_m2=left)
    with (output/"comparison.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps({"reflection_error_m":error,"full_sector_states":int(full.sum()),"relations":relation}),flush=True)


if __name__ == "__main__":
    main()

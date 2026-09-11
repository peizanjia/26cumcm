"""Train four symmetry networks and optimize four constant policies jointly."""
import argparse
import csv
import json
from pathlib import Path
import time
import numpy as np
import torch
from ..environment import make_dataset
from ..objective import outcomes, subset, evaluate, tensor_dataset
from .policy import TwoSidedNet

ROOT = Path(__file__).parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=21260911)
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(4)
    output, ckpts = ROOT/"results"/"symmetry", ROOT/"checkpoints"/"symmetry"
    output.mkdir(exist_ok=True); ckpts.mkdir(exist_ok=True)
    started = time.perf_counter()
    train = tensor_dataset(make_dataset(32768, 96, args.seed), device)
    val = tensor_dataset(make_dataset(1024, 128, args.seed+100000), device)
    nets = torch.nn.ModuleList([TwoSidedNet(args.seed+i, boundary_scaled=i>=2) for i in range(4)]).to(device)
    # Four independent starting points; no test data are used in optimization.
    fixed = torch.nn.Parameter(torch.tensor([[.5, .25], [.65, .18], [.8, .35], [.35, .15]], device=device))
    optimizer = torch.optim.Adam([{"params": nets.parameters(), "lr": 8e-4},
                                   {"params": [fixed], "lr": 2e-3}])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.steps, eta_min=2e-5)
    best = np.full(4, np.inf); best_fixed = np.inf
    history = []
    for step in range(args.steps+1):
        if step % 300 == 0 or step == args.steps:
            for i, net in enumerate(nets):
                branch_scores = []
                for side in (0, 1):
                    r = evaluate(val, lambda d, n=net, k=side: n(d["states"][:, 3:5])[:, k])
                    branch_scores.append(float(r["loss"].mean()))
                score = float(np.mean(branch_scores))
                history.append(dict(step=step, model=i, boundary_scaled=i>=2, mean_m2=score,
                                    right_m2=branch_scores[0], left_m2=branch_scores[1]))
                if score < best[i]:
                    best[i] = score
                    torch.save(dict(state_dict=net.state_dict(), boundary_scaled=i>=2,
                                    validation_m2=score, seed=args.seed+i, step=step), ckpts/f"model_{i}.pt")
            constants = []
            for i in range(4):
                # Use the right branch, matching the original network's branch.
                q = fixed[i].detach()*fixed.new_tensor([1., -1.])
                r = evaluate(val, lambda d, q=q: q.expand(len(d["states"]), -1))
                score = float(r["loss"].mean())
                constants.append(score)
                if score < best_fixed:
                    best_fixed = score
                    fixed_record = dict(a_m=float(q[0]*1500), abs_b_m=float(-q[1]*1500),
                                        validation_m2=score, step=step, start=i)
                    (output/"optimized_fixed.json").write_text(json.dumps(fixed_record, indent=2))
            print(json.dumps(dict(step=step, best_networks=best.tolist(), fixed=constants,
                                  elapsed_s=time.perf_counter()-started)), flush=True)
            with (output/"training.csv").open("w", newline="") as f:
                w=csv.DictWriter(f, fieldnames=list(history[0])); w.writeheader(); w.writerows(history)
        if step == args.steps:
            break
        indices = torch.randint(len(train["states"]), (192,), device=device)
        samples = torch.randint(96, (24,), device=device)
        d = subset(train, indices, samples)
        actions = [n(d["states"][:, 3:5])[:, k] for n in nets for k in (0, 1)]
        actions += [(q*fixed.new_tensor([1., -1.])).expand(len(indices), -1) for q in fixed]
        q = torch.cat(actions)
        dd = {k: v.repeat((len(actions),)+(1,)*(v.ndim-1)) for k, v in d.items()}
        losses = outcomes(dd, q)["loss"].reshape(len(actions), -1).mean(1)
        optimizer.zero_grad(set_to_none=True)
        (losses.sum()/1000).backward()
        torch.nn.utils.clip_grad_norm_([*nets.parameters(), fixed], 10.)
        optimizer.step(); scheduler.step()
        with torch.no_grad():
            fixed[:, 0].clamp_(-2, 2); fixed[:, 1].clamp_(.003, 2)
    selected = int(best.argmin())
    torch.save(torch.load(ckpts/f"model_{selected}.pt", weights_only=True), ckpts/"best.pt")
    summary = dict(seed=args.seed, validation_seed=args.seed+100000, states=32768, posterior=96,
                   batch_states=192, batch_samples=24, steps=args.steps, selected=selected,
                   best_networks_m2=best.tolist(), optimized_fixed=fixed_record,
                   elapsed_s=time.perf_counter()-started, gpu=torch.cuda.get_device_name(0) if device=="cuda" else None,
                   peak_gpu_memory_mb=torch.cuda.max_memory_allocated()/2**20 if device=="cuda" else 0,
                   objective="equal-weight mean of both side losses; side is external, never a true-target choice")
    (output/"training_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()

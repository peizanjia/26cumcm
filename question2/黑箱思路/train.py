"""Train independent networks together on one GPU, with fresh MC minibatches."""
import argparse
import csv
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from .environment import make_dataset
from .policy import PolicyNet
from .objective import tensor_dataset, subset, outcomes, evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--states", type=int, default=24000)
    parser.add_argument("--posterior", type=int, default=96)
    parser.add_argument("--batch", type=int, default=192)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--models", type=int, default=4)
    parser.add_argument("--validate-every", type=int, default=250)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    torch.set_num_threads(4)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    output = args.output
    (output / "results").mkdir(parents=True, exist_ok=True)
    (output / "checkpoints").mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    config = vars(args).copy()
    config.update(output=str(output), torch=torch.__version__, numpy=np.__version__,
                  cpu_count=os.cpu_count(), gpu=torch.cuda.get_device_name(0) if args.device == "cuda" else None,
                  objective="mean residual area m2; no_signal retains first area; near after optical=0",
                  train_seed=args.seed, validation_seed=args.seed + 100000,
                  arena_sides=256, hidden_radius_prior="Uniform(1000,1500)")
    (output / "results" / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print("Generating independent training and validation datasets", flush=True)
    train = tensor_dataset(make_dataset(args.states, args.posterior, args.seed), args.device)
    val = tensor_dataset(make_dataset(1024, 128, args.seed + 100000), args.device)
    print(f"Data ready in {time.perf_counter()-started:.1f}s; vertices={train['polygons'].shape[1]}", flush=True)
    models = torch.nn.ModuleList([PolicyNet(args.seed + i, 1 if i % 2 == 0 else -1)
                                  for i in range(args.models)]).to(args.device)
    optimizer = torch.optim.Adam(models.parameters(), lr=8e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.steps, eta_min=4e-5)
    best = np.full(args.models, np.inf)
    best_step = np.zeros(args.models, int)
    all_rows = []
    # All replicas share samples for lower-variance comparison but have
    # independent parameters and separate losses, with no cross-model mixing.
    for step in range(args.steps + 1):
        if step % args.validate_every == 0 or step == args.steps:
            for i, model in enumerate(models):
                result = evaluate(val, lambda d, m=model: m(d["states"][:, 3:5]))
                score = float(result["loss"].mean())
                row = dict(step=step, model=i, mean_area_m2=score,
                           no_signal=float(result["no_signal"].mean()),
                           elapsed_s=time.perf_counter() - started)
                all_rows.append(row)
                if score < best[i]:
                    best[i], best_step[i] = score, step
                    torch.save(dict(state_dict=model.state_dict(), step=step,
                                    validation_area_m2=score, seed=args.seed + i),
                               output / "checkpoints" / f"model_{i}.pt")
                print(json.dumps(row), flush=True)
            with (output / "results" / "training.csv").open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(all_rows[0]))
                w.writeheader()
                w.writerows(all_rows)
        if step == args.steps:
            break
        indices = torch.randint(args.states, (args.batch,), device=args.device)
        samples = torch.randint(args.posterior, (args.samples,), device=args.device)
        data = subset(train, indices, samples)
        # Merge model and state dimensions for one large GPU geometry kernel
        # workload; small networks themselves run independently.
        q = torch.cat([model(data["states"][:, 3:5]) for model in models])
        expanded = {key: value.repeat((args.models,) + (1,) * (value.ndim - 1))
                    for key, value in data.items()}
        losses = outcomes(expanded, q)["loss"].reshape(args.models, -1).mean(1)
        optimizer.zero_grad(set_to_none=True)
        (losses.sum() / 1000).backward()
        torch.nn.utils.clip_grad_norm_(models.parameters(), 10.)
        optimizer.step()
        scheduler.step()
        if step % 50 == 0:
            print(f"step={step} train_m2={losses.detach().cpu().tolist()}", flush=True)
    selected = int(np.argmin(best))
    checkpoint = torch.load(output / "checkpoints" / f"model_{selected}.pt", weights_only=True)
    torch.save(checkpoint, output / "checkpoints" / "best.pt")
    summary = dict(selected_model=selected, best_validation_m2=best.tolist(),
                   best_steps=best_step.tolist(), elapsed_s=time.perf_counter()-started,
                   peak_gpu_memory_mb=torch.cuda.max_memory_allocated() / 2**20 if args.device == "cuda" else 0,
                   status="completed")
    (output / "results" / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()

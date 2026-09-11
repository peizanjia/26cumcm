"""Held-out, paired policy comparisons and prior sensitivity experiments."""
import argparse
import csv
import json
from pathlib import Path
import time
import numpy as np
import torch
from .environment import make_dataset
from .geometry import SCALE, ARENA, first_polygon, second_cpu, area_cpu, robust_margin
from .policy import NeuralPolicy, certify_action
from .analytic_baseline import moment_action
from .objective import tensor_dataset, evaluate


def summarize(name, result, data):
    state_means = result["loss"].mean(axis=1).astype(float)
    se = state_means.std(ddof=1) / np.sqrt(len(state_means))
    q = result["q"]
    return dict(policy=name, mean_area_m2=float(state_means.mean()),
                ci95_low=float(state_means.mean()-1.96*se), ci95_high=float(state_means.mean()+1.96*se),
                radius_marginalized_loss_p50_m2=float(np.median(result["loss"])),
                radius_marginalized_loss_p95_m2=float(np.quantile(result["loss"], .95)),
                no_signal_probability=float(result["no_signal"].mean()),
                near_probability=float(result["near"].mean()),
                mean_move_m=float(np.linalg.norm(q, axis=1).mean()*SCALE),
                outside_arena_fraction=float(np.mean(np.linalg.norm(q+data["states"][:, 3:5], axis=1)>ARENA)))


def baselines(net, data):
    def fixed(a, b):
        return lambda d: torch.tensor([a/SCALE, b/SCALE], device=d["states"].device).expand(len(d["states"]), -1)
    def centroid_side(d):
        # Baseline uses a deterministic approximation to posterior mean from
        # the first feasible polygon, not held-out posterior/test targets.
        verts, counts = d["polygons"], d["counts"]
        index = torch.arange(verts.shape[1], device=verts.device)[None]
        end = (verts[..., 0] * (index < counts[:, None])).max(dim=1).values
        return torch.stack((.60 * end, .28 * end), dim=-1)
    methods = dict(neural=lambda d: net(d["states"][:, 3:5]),
                   repeat=fixed(0, 0), perpendicular_500=fixed(0, 500),
                   forward_750=fixed(750, 0), fixed_750_375=fixed(750, 375),
                   adaptive_60_28=centroid_side)
    def certified(d):
        q = net(d["states"][:, 3:5]).detach().cpu().numpy()
        p = d["states"][:, 3:5].cpu().numpy()
        safe = np.array([certify_action(pp, qq) for pp, qq in zip(p, q)])
        return torch.tensor(safe, device=d["states"].device, dtype=torch.float32)
    methods["neural_certified"] = certified
    methods["analytic_moments"] = lambda d: torch.tensor(
        moment_action(d["states"][:, 3:5].cpu().numpy()), device=d["states"].device, dtype=torch.float32)
    return methods


def geometry_audit(data, result, n=256):
    """Independent question1 half-plane implementation, finer arena, diameter.

    Full two-wedge polygon statistics are kept separate from the prior-clipped
    training region. Cases without a second direction are excluded explicitly.
    """
    from question1.geometry import intersect_bearings, polygon_diameter, polygon_area, _clip
    from .circle_reference import exact_circle_area
    rng = np.random.default_rng(99951)
    relative, absolute, diameters, pure_areas = [], [], [], []
    circle_errors = []
    unbounded, visible, contained, robust = 0, 0, 0, 0
    for i in range(min(n, len(data["states"]))):
        q = result["q"][i].astype(float)
        poly = first_polygon(data["states"][i, 3:5], 2048)
        robust += robust_margin(poly, q) >= 0
        # Actual hidden radius stays fixed between the two measurements.
        for j in range(4):
            g = data["targets"][i, j]
            lower = max(1000/SCALE, np.linalg.norm(g))
            radius = rng.uniform(lower, 1.)
            d2 = np.linalg.norm(g-q)
            if not 5/SCALE < d2 <= radius:
                continue
            visible += 1
            theta2 = np.arctan2((g-q)[1], (g-q)[0]) + data["errors"][i, j]
            # Use question1's independently implemented clipping routine.
            ref = poly.copy()
            for sign in [-1, 1]:
                a = theta2 + sign * np.pi/180
                normal = np.array([-sign*np.sin(a), sign*np.cos(a)])
                ref = _clip(ref, normal, float(np.dot(normal, q)), 1e-12)
            ref_area = polygon_area(ref)*SCALE**2
            approx = float(result["area"][i, j])
            absolute.append(abs(approx-ref_area))
            if i < 64:
                circle_area = exact_circle_area(data["states"][i, 3:5], q, theta2)*SCALE**2
                circle_errors.append(abs(approx-circle_area))
            relative.append(abs(approx-ref_area)/max(ref_area, 1.))
            diameter, _ = polygon_diameter(ref*SCALE)
            diameters.append(diameter)
            edges = np.roll(ref, -1, axis=0)-ref
            contained += bool(np.all(edges[:, 0]*(g-ref)[:, 1]-edges[:, 1]*(g-ref)[:, 0]>=-1e-8))
            full = intersect_bearings([[0, 0], q*SCALE], [0, np.degrees(theta2)])
            if full.status == "unbounded":
                unbounded += 1
            elif full.status == "bounded":
                pure_areas.append(polygon_area(full.vertices))
            else:
                raise AssertionError("Truth-consistent bearings cannot have empty intersection")
    return dict(states=min(n, len(data["states"])), direction_events=visible,
                truth_containment_rate=contained/max(visible, 1),
                independent_cpu_finer_arena_mean_abs_error_m2=float(np.mean(absolute)),
                independent_cpu_finer_arena_max_abs_error_m2=float(np.max(absolute)),
                independent_cpu_finer_arena_p95_relative_error=float(np.quantile(relative, .95)),
                clipped_diameter_p50_m=float(np.median(diameters)),
                clipped_diameter_p95_m=float(np.quantile(diameters, .95)),
                pure_wedges_unbounded_count=unbounded,
                pure_wedges_mean_area_when_bounded_m2=float(np.mean(pure_areas)),
                exact_circle_audit_events=len(circle_errors),
                exact_circle_mean_abs_error_m2=float(np.mean(circle_errors)),
                exact_circle_max_abs_error_m2=float(np.max(circle_errors)),
                conservative_1000m_certificate_fraction=robust/min(n, len(data["states"])))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--states", type=int, default=8192)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20460911)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    torch.set_num_threads(4)
    root = Path(__file__).parent
    output = root / "results"
    started = time.perf_counter()
    policy = NeuralPolicy(device=args.device)
    data = make_dataset(args.states, args.samples, args.seed)
    tensors = tensor_dataset(data, args.device)
    results, summaries = {}, []
    for name, action in baselines(policy.net, data).items():
        result = evaluate(tensors, action)
        results[name] = result
        row = summarize(name, result, data)
        summaries.append(row)
        print(json.dumps(row), flush=True)
    neural = results["neural"]["loss"].mean(1)
    paired = {}
    for name, result in results.items():
        delta = result["loss"].mean(1)-neural
        se = delta.std(ddof=1)/np.sqrt(len(delta))
        paired[name] = dict(baseline_minus_neural_m2=float(delta.mean()),
                            ci95_low=float(delta.mean()-1.96*se), ci95_high=float(delta.mean()+1.96*se),
                            relative_improvement=float(delta.mean()/result["loss"].mean()))
    audit = geometry_audit(data, results["neural"])
    print("geometry_audit", json.dumps(audit), flush=True)
    sensitivity = []
    for noise_kind, radius_mode in [("truncnorm", "uniform"), ("uniform", "1000"), ("uniform", "1500")]:
        test = make_dataset(2048, 128, args.seed+1000+len(sensitivity), noise_kind=noise_kind, radius_mode=radius_mode)
        td = tensor_dataset(test, args.device)
        for name, action in baselines(policy.net, test).items():
            if name not in ("neural", "fixed_750_375", "adaptive_60_28", "analytic_moments"):
                continue
            result = evaluate(td, action, radius_mode=radius_mode)
            row = summarize(name, result, test)
            row.update(noise=noise_kind, radius=radius_mode)
            sensitivity.append(row)
            print("sensitivity", json.dumps(row), flush=True)
    with (output / "comparison.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summaries[0])); w.writeheader(); w.writerows(summaries)
    np.savez_compressed(output / "heldout_state_results.npz", states=data["states"],
                        **{f"{name}_mean_area_m2": result["loss"].mean(1) for name, result in results.items()},
                        neural_q_local=results["neural"]["q"])
    report = dict(test_seed=args.seed, states=args.states, samples_per_state=args.samples,
                  independent_state_cluster_ci=True, summary=summaries, paired=paired,
                  geometry_audit=audit, sensitivity=sensitivity,
                  elapsed_s=time.perf_counter()-started)
    (output / "evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Evaluation completed", flush=True)


if __name__ == "__main__":
    main()

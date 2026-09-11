"""Finite-grid candidate region and a conservative reception certificate.

The 5% sublevel region is an MC/grid estimate, not a globally optimality
certificate. Output includes every evaluated point, not only a bounding box.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import torch
from .environment import sample_posterior, noise
from .geometry import SCALE, first_polygon, area_cpu, robust_margin
from .policy import NeuralPolicy
from .objective import tensor_dataset, outcomes


def state_data(x, y, theta_deg, samples, seed):
    theta = np.deg2rad(theta_deg % 360)
    c, s = np.cos(theta), np.sin(theta)
    local_p = np.array([c*x+s*y, -s*x+c*y])/SCALE
    poly = first_polygon(local_p, 1024)
    targets = sample_posterior(local_p[None], samples, seed)
    data = dict(states=np.array([[x/SCALE, y/SCALE, theta, *local_p]]),
                polygons=poly[None], counts=np.array([len(poly)]),
                areas=np.array([area_cpu(poly)]), targets=targets,
                errors=noise(np.random.default_rng(seed+1), (1, samples)))
    return data, local_p, poly


@torch.inference_mode()
def score_grid(data, actions, device="cuda", batch=64):
    td = tensor_dataset(data, device)
    scores, failures, ses = [], [], []
    for start in range(0, len(actions), batch):
        q = torch.tensor(actions[start:start+batch], dtype=torch.float32, device=device)
        d = {key: value.expand((len(q),) + value.shape[1:]) for key, value in td.items()}
        result = outcomes(d, q)
        scores.extend(result["loss"].mean(1).cpu().numpy())
        failures.extend(result["no_signal"].mean(1).cpu().numpy())
        ses.extend((result["loss"].std(1)/np.sqrt(data["targets"].shape[1])).cpu().numpy())
    return np.array(scores), np.array(failures), np.array(ses)


def analyze(x, y, theta, size=101, samples=2048, seed=6201, device="cuda", tag="custom"):
    root = Path(__file__).parent / "results" / "candidates"
    root.mkdir(parents=True, exist_ok=True)
    data, local_p, poly = state_data(x, y, theta, samples, seed)
    # Broad search covers every direction in the network's action box.
    coarse_axis = np.linspace(-3000, 3000, 61)
    ca, cb = np.meshgrid(coarse_axis, coarse_axis)
    coarse = np.column_stack((ca.ravel(), cb.ravel()))/SCALE
    cs, _, _ = score_grid(data, coarse, device)
    # A denser regional grid explicitly retains both lateral branches. These
    # limits are a declared computational window, not a rule in the problem.
    xa, ya = np.linspace(-250, 1750, size), np.linspace(-1000, 1000, size)
    xx, yy = np.meshgrid(xa, ya)
    actions = np.column_stack((xx.ravel(), yy.ravel()))/SCALE
    score, failure, se = score_grid(data, actions, device)
    policy = NeuralPolicy(device=device)
    with torch.inference_mode():
        nq = policy.net(torch.tensor(local_p, dtype=torch.float32, device=device)).cpu().numpy()
    nn_score, nn_fail, nn_se = score_grid(data, nq[None], device)
    best = int(np.argmin(score))
    best_score = min(float(cs.min()), float(score.min()), float(nn_score[0]))
    candidate = score <= 1.05*best_score
    margins = np.array([robust_margin(poly, q)*SCALE for q in actions])
    robust = margins >= 0
    # Independent posterior/error draws validate selected points, without
    # reusing the random samples that chose the grid minimum.
    check, _, _ = state_data(x, y, theta, samples*4, seed+100000)
    check_score, check_fail, check_se = score_grid(check, np.stack([nq, actions[best]]), device)
    angle = np.deg2rad(theta)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    world = np.einsum("ij,kj->ki", rotation, actions*SCALE)+[x, y]
    nn_world = np.einsum("ij,j->i", rotation, nq*SCALE)+[x, y]
    with (root / f"{tag}.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["x_m", "y_m", "along_m", "lateral_m", "expected_area_m2", "mc_se_m2",
                         "no_signal_probability", "within_5pct", "guaranteed_reception", "certificate_margin_m"])
        for i in range(len(actions)):
            writer.writerow([*world[i], *(actions[i]*SCALE), score[i], se[i], failure[i],
                             bool(candidate[i]), bool(robust[i]), margins[i]])
    result = dict(first_point=[x, y], bearing_deg=theta, grid_size=size,
                  grid_step_m=2000/(size-1), posterior_samples=samples, seed=seed,
                  broad_grid_best_m2=float(cs.min()), regional_grid_best_m2=float(score.min()),
                  nn_local_displacement_m=(nq*SCALE).tolist(), nn_second_point_m=nn_world.tolist(),
                  nn_in_sample_area_m2=float(nn_score[0]),
                  independent_validation=dict(samples=samples*4, nn_area_m2=float(check_score[0]),
                                              nn_se_m2=float(check_se[0]), grid_best_area_m2=float(check_score[1]),
                                              grid_best_se_m2=float(check_se[1]), nn_no_signal=float(check_fail[0])),
                  five_percent_grid_point_count=int(candidate.sum()),
                  guaranteed_reception_grid_point_count=int(robust.sum()),
                  both_grid_point_count=int((candidate & robust).sum()),
                  nn_1000m_certificate_margin_m=float(robust_margin(poly, nq.astype(float))*SCALE),
                  candidate_local_along_range_m=[float((actions[candidate, 0]*SCALE).min()),
                                                float((actions[candidate, 0]*SCALE).max())] if candidate.any() else None,
                  candidate_local_lateral_range_m=[float((actions[candidate, 1]*SCALE).min()),
                                                  float((actions[candidate, 1]*SCALE).max())] if candidate.any() else None,
                  caveat="5% region is a finite-grid MC estimate; coordinate extrema are not a filled rectangle")
    (root / f"{tag}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    np.savez_compressed(root / f"{tag}.npz", xx=xx, yy=yy, scores=score.reshape(xx.shape),
                        robust=robust.reshape(xx.shape), candidate=candidate.reshape(xx.shape),
                        nn=nq*SCALE, polygon=poly*SCALE, local_p=local_p*SCALE)
    print(json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--x", type=float, default=0)
    parser.add_argument("--y", type=float, default=0)
    parser.add_argument("--theta", type=float, default=0)
    parser.add_argument("--size", type=int, default=101)
    parser.add_argument("--samples", type=int, default=2048)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--examples", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.examples:
        cases = [(0, 0, 0, "origin_east"), (1400, 0, 180, "edge_inward"),
                 (1500, 0, 0, "edge_outward"), (800, 1000, 225, "oblique")]
        for i, (x, y, t, tag) in enumerate(cases):
            analyze(x, y, t, args.size, args.samples, 6201+i, args.device, tag)
    else:
        analyze(args.x, args.y, args.theta, args.size, args.samples, device=args.device)


if __name__ == "__main__":
    main()

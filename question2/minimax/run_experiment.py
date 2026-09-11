"""Paired simulation of mechanistic and repository neural policies."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .mechanistic import (
    ANGLE_ERROR, ARENA_RADIUS, NEAR_RADIUS, MechanisticPolicy,
    NeuralReferencePolicy, initial_region, point_in_convex, polygon_area,
    polygon_diameter, polygon_mec_radius, project_to_universal_safe,
    robust_score, second_region, universal_safe,
)


def sample_disk(rng: np.random.Generator) -> np.ndarray:
    angle = rng.uniform(-math.pi, math.pi)
    radius = ARENA_RADIUS * math.sqrt(rng.random())
    return radius * np.array([math.cos(angle), math.sin(angle)])


def sample_cases(n: int, seed: int) -> list[dict]:
    """Match the collaborator repository's stated synthetic test prior."""
    rng = np.random.default_rng(seed)
    cases = []
    while len(cases) < n:
        station, target = sample_disk(rng), sample_disk(rng)
        receive_radius = rng.uniform(1000.0, 1500.0)
        first_distance = float(np.linalg.norm(target - station))
        if not (NEAR_RADIUS < first_distance <= receive_radius):
            continue
        true_bearing = math.atan2(target[1] - station[1], target[0] - station[0])
        first_error = rng.uniform(-ANGLE_ERROR, ANGLE_ERROR)
        second_error = rng.uniform(-ANGLE_ERROR, ANGLE_ERROR)
        cases.append(dict(station=station, target=target,
                          receive_radius=receive_radius,
                          first_bearing=true_bearing + first_error,
                          second_error=second_error))
    return cases


def local_offset(station: np.ndarray, bearing: float, forward: float,
                 lateral: float) -> np.ndarray:
    c, s = math.cos(bearing), math.sin(bearing)
    return station + np.array([c * forward - s * lateral,
                               s * forward + c * lateral])


def outcome(first: np.ndarray, q: np.ndarray, target: np.ndarray,
            receive_radius: float, second_error: float,
            circle_sides: int) -> dict:
    first_area = polygon_area(first)
    first_diameter = polygon_diameter(first)
    first_radius = polygon_mec_radius(first)
    d2 = float(np.linalg.norm(target - q))
    if d2 <= NEAR_RADIUS:
        return dict(detected=1, near=1, area_m2=0.0, diameter_m=0.0,
                    mec_radius_m=0.0, repository_mec_radius_m=0.0,
                    legacy_area_m2=0.0, contains_truth=1)
    if d2 > receive_radius:
        return dict(detected=0, near=0, area_m2=first_area,
                    diameter_m=first_diameter, mec_radius_m=first_radius,
                    repository_mec_radius_m=first_radius, legacy_area_m2=first_area,
                    contains_truth=int(point_in_convex(first, target, 1e-5)))
    theta2 = math.atan2(target[1] - q[1], target[0] - q[0]) + second_error
    official = second_region(first, q, theta2, circle_sides, True)
    legacy = second_region(first, q, theta2, circle_sides, False)
    return dict(detected=1, near=0, area_m2=polygon_area(official),
                diameter_m=polygon_diameter(official),
                mec_radius_m=polygon_mec_radius(official),
                repository_mec_radius_m=polygon_mec_radius(legacy),
                legacy_area_m2=polygon_area(legacy),
                contains_truth=int(point_in_convex(official, target, 1e-5)))


def confidence_interval(values: np.ndarray) -> tuple[float, float]:
    if len(values) < 2:
        return float(values.mean()), float(values.mean())
    se = float(values.std(ddof=1) / math.sqrt(len(values)))
    mean = float(values.mean())
    return mean - 1.96 * se, mean + 1.96 * se


def summarize(rows: list[dict]) -> list[dict]:
    result = []
    policies = sorted({row["policy"] for row in rows})
    for name in policies:
        part = [row for row in rows if row["policy"] == name]
        area = np.array([row["area_m2"] for row in part])
        diameter = np.array([row["diameter_m"] for row in part])
        radius = np.array([row["mec_radius_m"] for row in part])
        repository_radius = np.array(
            [row["repository_mec_radius_m"] for row in part])
        legacy = np.array([row["legacy_area_m2"] for row in part])
        conditional_worst = np.array(
            [row["conditional_worst_diameter_m"] for row in part])
        ci_lo, ci_hi = confidence_interval(area)
        result.append(dict(
            policy=name,
            trials=len(part),
            mean_area_m2=float(area.mean()),
            area_ci95_low_m2=ci_lo,
            area_ci95_high_m2=ci_hi,
            p95_area_m2=float(np.quantile(area, 0.95)),
            mean_diameter_m=float(diameter.mean()),
            p95_diameter_m=float(np.quantile(diameter, 0.95)),
            max_diameter_m=float(diameter.max()),
            mean_mec_radius_m=float(radius.mean()),
            p95_mec_radius_m=float(np.quantile(radius, 0.95)),
            max_mec_radius_m=float(radius.max()),
            mec_mae20_m=float(np.abs(radius - 20.0).mean()),
            mec_success_le20=float(np.mean(radius <= 20.0)),
            mean_repository_mec_radius_m=float(repository_radius.mean()),
            p95_repository_mec_radius_m=float(np.quantile(repository_radius, 0.95)),
            max_repository_mec_radius_m=float(repository_radius.max()),
            repository_mec_mae20_m=float(np.abs(repository_radius - 20.0).mean()),
            repository_mec_success_le20=float(np.mean(repository_radius <= 20.0)),
            mean_legacy_area_m2=float(legacy.mean()),
            no_signal_rate=float(1.0 - np.mean([row["detected"] for row in part])),
            certified_reception_rate=float(np.mean([row["safe_certified"] for row in part])),
            mean_conditional_worst_diameter_m=float(conditional_worst.mean()),
            p95_conditional_worst_diameter_m=float(np.quantile(conditional_worst, 0.95)),
            conservative_mean_worst_diameter_m=float(np.mean(
                [row["conservative_worst_diameter_m"] for row in part])),
            conservative_max_worst_diameter_m=float(np.max(
                [row["conservative_worst_diameter_m"] for row in part])),
            mean_move_m=float(np.mean([row["move_m"] for row in part])),
            mean_decision_time_ms=float(np.mean([row["decision_time_ms"] for row in part])),
            containment_rate=float(np.mean([row["contains_truth"] for row in part])),
        ))
    return result


def plot_summary(summary: list[dict], output: Path) -> None:
    labels = [item["policy"] for item in summary]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    axes[0].bar(x, [item["mean_area_m2"] for item in summary], color="#4c78a8")
    axes[0].set_ylabel("Mean residual area (m²)")
    axes[0].set_title("Average-case geometry")
    axes[1].bar(x, [item["p95_diameter_m"] for item in summary], color="#f58518")
    axes[1].set_ylabel("95th percentile diameter (m)")
    axes[1].set_title("Observed tail")
    axes[2].bar(x, [item["conservative_mean_worst_diameter_m"] for item in summary],
                color="#54a24b")
    axes[2].set_ylabel("Mean robust diameter bound (m)")
    axes[2].set_title("Pre-measurement robustness")
    for axis in axes:
        axis.set_xticks(x, labels, rotation=28, ha="right")
        axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_representative(policy: MechanisticPolicy, neural: NeuralReferencePolicy,
                        output: Path) -> dict:
    station = np.zeros(2)
    bearing = 0.0
    first = initial_region(station, bearing, policy.circle_sides)
    decision = policy.decide(station, bearing)
    nn = np.array(neural(0.0, 0.0, 0.0))
    with output.with_name("candidate_points.csv").open(
            "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["x_m", "y_m", "worst_diameter_m", "worst_area_m2",
                         "within_5m_of_best"])
        for point, score, selected in zip(decision.candidate_points,
                                          decision.candidate_scores,
                                          decision.near_optimal_mask):
            writer.writerow([point[0], point[1], score[0], score[1], int(selected)])
    fig, axis = plt.subplots(figsize=(8.2, 5.8))
    scatter = axis.scatter(decision.candidate_points[:, 0], decision.candidate_points[:, 1],
                           c=decision.candidate_scores[:, 0], s=22, cmap="viridis_r")
    near = decision.candidate_points[decision.near_optimal_mask]
    axis.scatter(near[:, 0], near[:, 1], facecolors="none", edgecolors="#e45756",
                 s=55, linewidths=1.2, label="within 5 m of best robust diameter")
    axis.plot(decision.point[0], decision.point[1], "r*", ms=14, label="mechanistic optimum")
    axis.plot(nn[0], nn[1], "kx", ms=10, mew=2, label="repository neural action")
    axis.fill(first[:, 0], first[:, 1], color="#72b7b2", alpha=0.18,
              label="first feasible target set")
    axis.plot(first[:, 0], first[:, 1], color="#2a7774", lw=1)
    axis.axhline(0, color="0.75", lw=0.7)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("forward offset from first monitor (m)")
    axis.set_ylabel("lateral offset (m)")
    axis.set_title("Explainable candidate set, S₁=(0,0), first bearing=0°")
    axis.legend(loc="upper left", fontsize=8)
    fig.colorbar(scatter, ax=axis, label="worst post-measurement diameter (m)")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return dict(mechanistic_point_m=decision.point.tolist(),
                mechanistic_worst_diameter_m=decision.score.worst_diameter_m,
                mechanistic_worst_area_m2=decision.score.worst_area_m2,
                neural_point_m=nn.tolist(),
                neural_safe_certificate=universal_safe(station, bearing, nn),
                candidates=int(len(decision.candidate_points)),
                near_optimal_candidates=int(decision.near_optimal_mask.sum()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=96)
    parser.add_argument("--seed", type=int, default=20660911)
    parser.add_argument("--circle-sides", type=int, default=64)
    parser.add_argument("--angle-step-deg", type=float, default=2.0)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).parent / "results")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    local_checkpoint = Path(__file__).parent / "reference" / "original_best.pt"
    repository_checkpoint = (
        Path(__file__).parent.parent / "黑箱思路" / "checkpoints" / "best.pt"
    )
    checkpoint = (local_checkpoint if local_checkpoint.exists()
                  else repository_checkpoint)
    neural = NeuralReferencePolicy(checkpoint)
    mechanistic = MechanisticPolicy(args.circle_sides, args.angle_step_deg)
    cases = sample_cases(args.trials, args.seed)
    rows: list[dict] = []
    started = time.perf_counter()

    for trial, case in enumerate(cases):
        station = case["station"]
        bearing = case["first_bearing"]
        theta_deg = math.degrees(bearing)
        first = initial_region(station, bearing, args.circle_sides)
        first_diameter = polygon_diameter(first)

        t0 = time.perf_counter()
        decision = mechanistic.decide(station, bearing)
        mech_ms = (time.perf_counter() - t0) * 1000.0
        t0 = time.perf_counter()
        nn_point = np.array(neural(station[0], station[1], theta_deg))
        nn_ms = (time.perf_counter() - t0) * 1000.0
        projected = project_to_universal_safe(station, bearing, nn_point)

        points = {
            "mechanistic_minimax": (decision.point, mech_ms, decision.score),
            "repository_neural": (nn_point, nn_ms, None),
            "neural_safe_projection": (projected, nn_ms, None),
            "analytic_far_range": (local_offset(station, bearing, 12000/13, 5000/13), 0.0, None),
            "fixed_750_375": (local_offset(station, bearing, 750.0, 375.0), 0.0, None),
            "perpendicular_500": (local_offset(station, bearing, 0.0, 500.0), 0.0, None),
        }
        for name, (q, elapsed_ms, known_score) in points.items():
            observed = outcome(first, q, case["target"], case["receive_radius"],
                               case["second_error"], args.circle_sides)
            safe = universal_safe(station, bearing, q)
            score = known_score or robust_score(first, q, args.angle_step_deg,
                                                  args.circle_sides)
            conservative_worst = score.worst_diameter_m if safe else first_diameter
            rows.append(dict(trial=trial, policy=name,
                             station_x_m=float(station[0]), station_y_m=float(station[1]),
                             target_x_m=float(case["target"][0]), target_y_m=float(case["target"][1]),
                             q_x_m=float(q[0]), q_y_m=float(q[1]),
                             receive_radius_m=float(case["receive_radius"]),
                             move_m=float(np.linalg.norm(q - station)),
                             decision_time_ms=elapsed_ms,
                             safe_certified=int(safe),
                             conditional_worst_diameter_m=score.worst_diameter_m,
                             conditional_worst_area_m2=score.worst_area_m2,
                             conservative_worst_diameter_m=conservative_worst,
                             **observed))
        if (trial + 1) % 16 == 0:
            print(f"completed {trial + 1}/{args.trials}", flush=True)

    fields = list(rows[0])
    with (args.output / "trial_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize(rows)
    representative = plot_representative(mechanistic, neural,
                                           args.output / "candidate_region.png")
    plot_summary(summary, args.output / "comparison.png")
    payload = dict(config=dict(trials=args.trials, seed=args.seed,
                               circle_sides=args.circle_sides,
                               angle_step_deg=args.angle_step_deg,
                               synthetic_prior="S1,G uniform-area in arena; rho uniform[1000,1500]; condition 5<d1<=rho; errors uniform[-1,1] deg",
                               circle_geometry="circumscribed regular polygons (conservative)"),
                   summary=summary, representative=representative,
                   elapsed_s=time.perf_counter() - started)
    (args.output / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

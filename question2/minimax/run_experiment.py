"""Paired simulation of three interpretable second-monitor policies."""

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
    boundary_aware_safe, initial_region, point_in_convex,
    polygon_area, polygon_diameter, polygon_mec_radius, reception_margin_from_supports,
    reception_supports, robust_score, second_region,
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
        # Keep the same bearing-return cases for paired comparability, while
        # the decision model itself deliberately neglects the 5 m distinction.
        if not (NEAR_RADIUS < first_distance <= receive_radius):
            continue
        true_bearing = math.atan2(target[1] - station[1], target[0] - station[0])
        first_error = rng.uniform(-ANGLE_ERROR, ANGLE_ERROR)
        second_error = rng.uniform(-ANGLE_ERROR, ANGLE_ERROR)
        cases.append(dict(station=station, target=target,
                          receive_radius=receive_radius,
                          first_bearing=true_bearing + first_error,
                          second_error=second_error,
                          scenario="global_random"))
    return cases


def sample_edge_cases(n: int, seed: int) -> list[dict]:
    """Boundary-outward stress cases; this is not the global synthetic prior."""
    rng = np.random.default_rng(seed)
    cases = []
    while len(cases) < n:
        station_angle = rng.uniform(-math.pi, math.pi)
        station_radius = rng.uniform(1500.0, 1775.0)
        station = station_radius * np.array(
            [math.cos(station_angle), math.sin(station_angle)])
        offset = rng.uniform(-math.radians(70.0), math.radians(70.0))
        true_bearing = station_angle + offset
        direction = np.array([math.cos(true_bearing), math.sin(true_bearing)])
        projection = float(np.dot(station, direction))
        discriminant = (projection * projection + ARENA_RADIUS**2
                        - float(np.dot(station, station)))
        if discriminant <= 0:
            continue
        arena_limit = -projection + math.sqrt(discriminant)
        receive_radius = rng.uniform(1000.0, 1500.0)
        upper = min(arena_limit, receive_radius)
        if upper <= 25.0:
            continue
        target_distance = rng.uniform(10.0, upper)
        target = station + target_distance * direction
        first_error = rng.uniform(-ANGLE_ERROR, ANGLE_ERROR)
        second_error = rng.uniform(-ANGLE_ERROR, ANGLE_ERROR)
        cases.append(dict(station=station, target=target,
                          receive_radius=receive_radius,
                          first_bearing=true_bearing + first_error,
                          second_error=second_error,
                          scenario="edge_outward_stress"))
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
    if d2 > receive_radius:
        return dict(detected=0, near=0, area_m2=first_area,
                    diameter_m=first_diameter, mec_radius_m=first_radius,
                    contains_truth=int(point_in_convex(first, target, 1e-5)))
    theta2 = math.atan2(target[1] - q[1], target[0] - q[0]) + second_error
    official = second_region(first, q, theta2, circle_sides, True)
    return dict(detected=1, near=0, area_m2=polygon_area(official),
                diameter_m=polygon_diameter(official),
                mec_radius_m=polygon_mec_radius(official),
                contains_truth=int(point_in_convex(official, target, 1e-5)))


def confidence_interval(values: np.ndarray) -> tuple[float, float]:
    if len(values) < 2:
        return float(values.mean()), float(values.mean())
    se = float(values.std(ddof=1) / math.sqrt(len(values)))
    mean = float(values.mean())
    return mean - 1.96 * se, mean + 1.96 * se


def summarize(rows: list[dict]) -> list[dict]:
    result = []
    groups = sorted({(row["scenario"], row["policy"]) for row in rows})
    for scenario, name in groups:
        part = [row for row in rows
                if row["policy"] == name and row["scenario"] == scenario]
        area = np.array([row["area_m2"] for row in part])
        diameter = np.array([row["diameter_m"] for row in part])
        radius = np.array([row["mec_radius_m"] for row in part])
        conditional_worst = np.array(
            [row["conditional_worst_mec_radius_m"] for row in part])
        ci_lo, ci_hi = confidence_interval(area)
        result.append(dict(
            scenario=scenario,
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
            mec_success_le20=float(np.mean(radius <= 20.0)),
            no_signal_rate=float(1.0 - np.mean([row["detected"] for row in part])),
            certified_reception_rate=float(np.mean([row["safe_certified"] for row in part])),
            mean_conditional_worst_mec_radius_m=float(conditional_worst.mean()),
            p95_conditional_worst_mec_radius_m=float(np.quantile(conditional_worst, 0.95)),
            conservative_mean_worst_mec_radius_m=float(np.mean(
                [row["conservative_worst_mec_radius_m"] for row in part])),
            conservative_max_worst_mec_radius_m=float(np.max(
                [row["conservative_worst_mec_radius_m"] for row in part])),
            mean_move_m=float(np.mean([row["move_m"] for row in part])),
            mean_decision_time_ms=float(np.mean([row["decision_time_ms"] for row in part])),
            containment_rate=float(np.mean([row["contains_truth"] for row in part])),
        ))
    return result


def paired_comparisons(rows: list[dict]) -> list[dict]:
    """Paired baseline-minus-minimax differences; positive favours minimax."""
    output = []
    for scenario in sorted({row["scenario"] for row in rows}):
        part = [row for row in rows if row["scenario"] == scenario]
        trials = sorted({int(row["trial"]) for row in part})
        by_key = {(int(row["trial"]), row["policy"]): row for row in part}
        for baseline in ("analytic_far_range", "fixed_750_375"):
            observed = np.array([
                float(by_key[(trial, baseline)]["mec_radius_m"])
                - float(by_key[(trial, "minimax_mec")]["mec_radius_m"])
                for trial in trials
            ])
            robust = np.array([
                float(by_key[(trial, baseline)]["conservative_worst_mec_radius_m"])
                - float(by_key[(trial, "minimax_mec")]["conservative_worst_mec_radius_m"])
                for trial in trials
            ])
            observed_ci = confidence_interval(observed)
            robust_ci = confidence_interval(robust)
            output.append(dict(
                scenario=scenario, baseline=baseline, trials=len(trials),
                observed_radius_improvement_m=float(observed.mean()),
                observed_radius_improvement_ci95_m=list(observed_ci),
                robust_radius_improvement_m=float(robust.mean()),
                robust_radius_improvement_ci95_m=list(robust_ci),
            ))
    return output


def plot_summary(summary: list[dict], output: Path) -> None:
    scenarios = ["global_random", "edge_outward_stress"]
    policy_order = ["minimax_mec", "analytic_far_range", "fixed_750_375"]
    metrics = [
        ("mean_mec_radius_m", "Observed mean", "Mean MEC radius (m)", "#4c78a8"),
        ("p95_mec_radius_m", "Observed tail", "P95 MEC radius (m)", "#f58518"),
        ("conservative_mean_worst_mec_radius_m", "Conditional minimax score",
         "Mean robust MEC bound (m)", "#54a24b"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.2))
    for row_index, scenario in enumerate(scenarios):
        lookup = {item["policy"]: item for item in summary
                  if item["scenario"] == scenario}
        labels = [name for name in policy_order if name in lookup]
        x = np.arange(len(labels))
        for column_index, (field, title, ylabel, color) in enumerate(metrics):
            axis = axes[row_index, column_index]
            axis.bar(x, [lookup[name][field] for name in labels], color=color)
            if row_index == 0:
                axis.set_title(title)
            axis.set_ylabel(ylabel)
            axis.set_xticks(x, [name.replace("_", "\n") for name in labels])
            axis.grid(axis="y", alpha=0.25)
        axes[row_index, 0].text(
            -0.28, 0.5,
            "Global random" if scenario == "global_random" else "Edge-outward stress",
            transform=axes[row_index, 0].transAxes, rotation=90,
            va="center", ha="center", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _margin_grid(station: np.ndarray, bearing: float, xs: np.ndarray,
                 ys: np.ndarray, safe_step_deg: float) -> np.ndarray:
    supports = reception_supports(station, bearing, safe_step_deg)
    result = np.empty((len(ys), len(xs)))
    for row, y in enumerate(ys):
        points = np.column_stack((xs, np.full(len(xs), y)))
        distance = np.linalg.norm(
            points[:, None, :] - supports.points[None, :, :], axis=2)
        result[row] = np.min(supports.radii[None, :] - distance, axis=1)
    return result


def plot_representative(policy: MechanisticPolicy, output: Path) -> dict:
    cases = [
        ("center", np.array([0.0, 0.0]), 0.0),
        ("outward edge", np.array([1700.0, 0.0]), 0.0),
        ("near tangent", np.array([1700.0, 0.0]), math.pi / 2.0),
    ]
    decisions = []
    with output.with_name("candidate_points.csv").open(
            "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["scenario", "x_m", "y_m", "worst_mec_radius_m",
                         "worst_area_m2", "worst_diameter_m", "within_5pct",
                         "robust_le20"])
        for name, station, bearing in cases:
            decision = policy.decide(station, bearing)
            decisions.append((name, station, bearing, decision))
            for point, score, selected in zip(decision.candidate_points,
                                              decision.candidate_scores,
                                              decision.near_optimal_mask):
                writer.writerow([name, point[0], point[1], score[0], score[1],
                                 score[2], int(selected), int(score[0] <= 20.0)])

    fig, axes = plt.subplots(1, 3, figsize=(17.2, 5.5), constrained_layout=True)
    payload = {}
    for axis, (name, station, bearing, decision) in zip(axes, decisions):
        first = initial_region(station, bearing, policy.circle_sides)
        xs = np.linspace(station[0] - 1050.0, station[0] + 1050.0, 121)
        ys = np.linspace(station[1] - 1050.0, station[1] + 1050.0, 121)
        margin = _margin_grid(station, bearing, xs, ys, policy.safe_angle_step_deg)
        axis.contourf(xs, ys, margin, levels=[0.0, float(np.max(margin)) + 1.0],
                      colors=["#d8eee8"], alpha=0.65)
        axis.contour(xs, ys, margin, levels=[0.0], colors=["#238b73"], linewidths=1.5)
        scatter = axis.scatter(decision.candidate_points[:, 0],
                               decision.candidate_points[:, 1],
                               c=decision.candidate_scores[:, 0], s=15,
                               cmap="viridis_r", label="reliable candidates")
        selected = decision.candidate_points[decision.near_optimal_mask]
        threshold = decision.candidate_points[decision.candidate_scores[:, 0] <= 20.0]
        if len(threshold):
            axis.scatter(threshold[:, 0], threshold[:, 1], facecolors="none",
                         edgecolors="#3b5bdb", s=27, linewidths=0.8,
                         label="robust MEC ≤ 20 m")
        axis.scatter(selected[:, 0], selected[:, 1], facecolors="none",
                     edgecolors="#e45756", s=45, linewidths=1.0,
                     label="within 5% of best")
        axis.plot(decision.point[0], decision.point[1], "r*", ms=13,
                  label="minimax MEC")
        axis.fill(first[:, 0], first[:, 1], color="#72b7b2", alpha=0.28,
                  label="first target set")
        axis.plot(first[:, 0], first[:, 1], color="#2a7774", lw=1)
        axis.plot(station[0], station[1], "ko", ms=4, label="first monitor")
        arena = plt.Circle((0, 0), ARENA_RADIUS, fill=False, color="0.55",
                           linestyle="--", linewidth=0.9)
        axis.add_patch(arena)
        axis.set_xlim(xs[0], xs[-1])
        axis.set_ylim(ys[0], ys[-1])
        axis.set_aspect("equal", adjustable="box")
        axis.set_title(name)
        axis.set_xlabel("x (m)")
        axis.grid(alpha=0.15)
        fig.colorbar(scatter, ax=axis, shrink=0.76, label="worst MEC radius (m)")
        payload[name] = dict(point_m=decision.point.tolist(),
                             worst_mec_radius_m=decision.score.worst_mec_radius_m,
                             worst_diameter_m=decision.score.worst_diameter_m,
                             worst_area_m2=decision.score.worst_area_m2,
                             candidates=int(len(decision.candidate_points)),
                             near_optimal_candidates=int(decision.near_optimal_mask.sum()))
    axes[0].set_ylabel("y (m)")
    # The central example has no 20 m-feasible point, so add a proxy to keep
    # the blue-outline encoding visible in the shared legend.
    axes[0].scatter([], [], facecolors="none", edgecolors="#3b5bdb",
                    s=27, linewidths=0.8, label="robust MEC ≤ 20 m")
    axes[0].legend(loc="upper left", fontsize=7)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=48)
    parser.add_argument("--edge-trials", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20660911)
    parser.add_argument("--circle-sides", type=int, default=128)
    parser.add_argument("--angle-step-deg", type=float, default=2.0)
    parser.add_argument("--safe-angle-step-deg", type=float, default=0.05)
    parser.add_argument("--safe-guard-m", type=float, default=0.0)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).parent / "results_mec")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    mechanistic = MechanisticPolicy(
        args.circle_sides, args.angle_step_deg,
        safe_angle_step_deg=args.safe_angle_step_deg,
        safe_guard_m=args.safe_guard_m)
    cases = sample_cases(args.trials, args.seed)
    cases += sample_edge_cases(args.edge_trials, args.seed + 1)
    rows: list[dict] = []
    started = time.perf_counter()

    for trial, case in enumerate(cases):
        station = case["station"]
        bearing = case["first_bearing"]
        first = initial_region(station, bearing, args.circle_sides)
        first_radius = polygon_mec_radius(first)
        supports = reception_supports(station, bearing, args.safe_angle_step_deg)

        t0 = time.perf_counter()
        decision = mechanistic.decide(station, bearing)
        mech_ms = (time.perf_counter() - t0) * 1000.0
        points = {
            "minimax_mec": (decision.point, mech_ms, decision.score),
            "analytic_far_range": (local_offset(station, bearing, 12000/13, 5000/13), 0.0, None),
            "fixed_750_375": (local_offset(station, bearing, 750.0, 375.0), 0.0, None),
        }
        for name, (q, elapsed_ms, known_score) in points.items():
            observed = outcome(first, q, case["target"], case["receive_radius"],
                               case["second_error"], args.circle_sides)
            margin = reception_margin_from_supports(supports, q)
            safe = boundary_aware_safe(
                station, bearing, q, args.safe_angle_step_deg,
                args.safe_guard_m, supports)
            score = known_score or robust_score(first, q, args.angle_step_deg,
                                                  args.circle_sides)
            conservative_worst = score.worst_mec_radius_m if safe else first_radius
            rows.append(dict(trial=trial, policy=name,
                             scenario=case["scenario"],
                             station_x_m=float(station[0]), station_y_m=float(station[1]),
                             target_x_m=float(case["target"][0]), target_y_m=float(case["target"][1]),
                             q_x_m=float(q[0]), q_y_m=float(q[1]),
                             receive_radius_m=float(case["receive_radius"]),
                             move_m=float(np.linalg.norm(q - station)),
                             decision_time_ms=elapsed_ms,
                             safe_certified=int(safe),
                             safe_margin_m=margin,
                             conditional_worst_mec_radius_m=score.worst_mec_radius_m,
                             conditional_worst_diameter_m=score.worst_diameter_m,
                             conditional_worst_area_m2=score.worst_area_m2,
                             conservative_worst_mec_radius_m=conservative_worst,
                             **observed))
        if (trial + 1) % 8 == 0:
            print(f"completed {trial + 1}/{len(cases)}", flush=True)

    fields = list(rows[0])
    with (args.output / "trial_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize(rows)
    representative = plot_representative(
        mechanistic, args.output / "candidate_regions.png")
    plot_summary(summary, args.output / "comparison.png")
    payload = dict(config=dict(global_trials=args.trials,
                               edge_outward_trials=args.edge_trials, seed=args.seed,
                               circle_sides=args.circle_sides,
                               angle_step_deg=args.angle_step_deg,
                               safe_angle_step_deg=args.safe_angle_step_deg,
                               safe_guard_m=args.safe_guard_m,
                               synthetic_prior="S1,G uniform-area in arena; rho uniform[1000,1500]; condition 5<d1<=rho; errors uniform[-1,1] deg; policy neglects the 5 m distinction",
                               boundary_stress="S1 radius uniform[1500,1775], outward bearing offset within +/-70 deg, target sampled on the clipped ray",
                               circle_geometry="circumscribed regular polygons (conservative)"),
                   summary=summary, paired_comparisons=paired_comparisons(rows),
                   representative=representative,
                   elapsed_s=time.perf_counter() - started)
    (args.output / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

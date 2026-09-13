"""Generate paper-ready visual diagnostics from the completed MEC experiment."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

from .mechanistic import (
    MechanisticPolicy, clip_disk, clip_wedge, initial_region,
    polygon_min_enclosing_circle, robust_score,
)


POLICIES = ["minimax_mec", "analytic_far_range", "fixed_750_375"]
POLICY_LABELS = {
    "minimax_mec": "Minimax MEC",
    "analytic_far_range": "Far-field analytic",
    "fixed_750_375": "Fixed (750, 375)",
}
COLORS = {
    "minimax_mec": "#3b6ea8",
    "analytic_far_range": "#e28e2c",
    "fixed_750_375": "#59a14f",
}
SCENARIO_LABELS = {
    "global_random": "Global random",
    "edge_outward_stress": "Edge-outward stress",
}


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=210, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_distributions(rows: list[dict], output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 8.2), constrained_layout=True)
    rng = np.random.default_rng(20260912)
    for row_index, scenario in enumerate(("global_random", "edge_outward_stress")):
        part = [row for row in rows if row["scenario"] == scenario]
        values = [np.array([float(row["mec_radius_m"]) for row in part
                            if row["policy"] == policy]) for policy in POLICIES]

        axis = axes[row_index, 0]
        violin = axis.violinplot(values, showmedians=True, showextrema=True)
        for body, policy in zip(violin["bodies"], POLICIES):
            body.set_facecolor(COLORS[policy])
            body.set_edgecolor("black")
            body.set_alpha(0.62)
        for key in ("cmedians", "cbars", "cmins", "cmaxes"):
            violin[key].set_color("#333333")
        for index, (policy, data) in enumerate(zip(POLICIES, values), start=1):
            jitter = rng.normal(0.0, 0.035, len(data))
            axis.scatter(index + jitter, data, s=11, alpha=0.55,
                         color=COLORS[policy], edgecolor="none")
        axis.axhline(20.0, color="#c43c39", linestyle="--", linewidth=1.2,
                     label="20 m threshold")
        axis.set_xticks(range(1, 4), [POLICY_LABELS[p] for p in POLICIES],
                        rotation=10)
        axis.set_ylabel("Observed MEC radius (m)")
        axis.set_title(f"{SCENARIO_LABELS[scenario]}: distribution")
        axis.grid(axis="y", alpha=0.2)
        axis.legend(frameon=False, fontsize=8)

        axis = axes[row_index, 1]
        for policy, data in zip(POLICIES, values):
            ordered = np.sort(data)
            probability = np.arange(1, len(ordered) + 1) / len(ordered)
            axis.step(ordered, probability, where="post", linewidth=2,
                      color=COLORS[policy], label=POLICY_LABELS[policy])
        axis.axvline(20.0, color="#c43c39", linestyle="--", linewidth=1.2)
        axis.set_xlabel("Observed MEC radius (m)")
        axis.set_ylabel("Empirical cumulative probability")
        axis.set_ylim(0.0, 1.02)
        axis.set_title(f"{SCENARIO_LABELS[scenario]}: ECDF")
        axis.grid(alpha=0.2)
        axis.legend(frameon=False, fontsize=8, loc="lower right")
    save(fig, output)


def plot_paired(rows: list[dict], output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 9.4), constrained_layout=True)
    baselines = ("analytic_far_range", "fixed_750_375")
    for row_index, scenario in enumerate(("global_random", "edge_outward_stress")):
        part = [row for row in rows if row["scenario"] == scenario]
        lookup = {(int(row["trial"]), row["policy"]): float(row["mec_radius_m"])
                  for row in part}
        trials = sorted({int(row["trial"]) for row in part})
        for column_index, baseline in enumerate(baselines):
            axis = axes[row_index, column_index]
            x = np.array([lookup[(trial, "minimax_mec")] for trial in trials])
            y = np.array([lookup[(trial, baseline)] for trial in trials])
            improvement = y - x
            extent = [float(min(x.min(), y.min())), float(max(x.max(), y.max()))]
            padding = max(2.0, 0.06 * (extent[1] - extent[0]))
            lo, hi = extent[0] - padding, extent[1] + padding
            scatter = axis.scatter(x, y, c=improvement, cmap="RdYlGn",
                                   vmin=-max(abs(improvement.min()), abs(improvement.max())),
                                   vmax=max(abs(improvement.min()), abs(improvement.max())),
                                   s=35, edgecolor="white", linewidth=0.45)
            axis.plot([lo, hi], [lo, hi], color="0.35", linestyle="--", linewidth=1)
            axis.fill_between([lo, hi], [lo, hi], [hi, hi], color="#59a14f",
                              alpha=0.07)
            axis.set_xlim(lo, hi)
            axis.set_ylim(lo, hi)
            axis.set_aspect("equal", adjustable="box")
            axis.set_xlabel("Minimax observed MEC radius (m)")
            axis.set_ylabel(f"{POLICY_LABELS[baseline]} radius (m)")
            axis.set_title(
                f"{SCENARIO_LABELS[scenario]}\nmean improvement = {improvement.mean():.2f} m")
            axis.grid(alpha=0.18)
            fig.colorbar(scatter, ax=axis, shrink=0.77,
                         label="baseline minus Minimax (m)")
    save(fig, output)


def plot_safe_area(payload: dict, output: Path) -> None:
    cases = payload["cases"]
    labels = [case["scenario"] for case in cases]
    legacy = np.array([case["legacy_area_km2"] for case in cases])
    recovered = np.array([case["recovered_area_km2"] for case in cases])
    complete = legacy + recovered
    x = np.arange(len(cases))
    fig, axis = plt.subplots(figsize=(9.6, 5.2), constrained_layout=True)
    axis.bar(x, legacy, color="#59b3a5", label="Legacy safe subset")
    axis.bar(x, recovered, bottom=legacy, color="#ed8585",
             label="Recovered by boundary-aware set")
    for index, (total, fraction) in enumerate(zip(
            complete, [case["legacy_fraction_percent"] for case in cases])):
        axis.text(index, total + 0.055, f"{total:.3f} km²\nlegacy {fraction:.1f}%",
                  ha="center", va="bottom", fontsize=9)
    axis.set_xticks(x, labels)
    axis.set_ylabel("Reliable reception area (km²)")
    axis.set_ylim(0, complete.max() * 1.22)
    axis.set_title("How much admissible area the legacy safe subset discards")
    axis.grid(axis="y", alpha=0.2)
    axis.legend(frameon=False, ncols=2, loc="upper left")
    save(fig, output)


def plot_convergence(payload: dict, output: Path) -> None:
    audit = payload["audit"]
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.2), constrained_layout=True)
    labels = {
        "center_minimax": "Center Minimax",
        "edge_outward": "Edge-outward Minimax",
        "analytic_far_range": "Center far-field",
    }
    colors = dict(zip(labels, ("#3b6ea8", "#59a14f", "#e28e2c")))
    for key in labels:
        data = sorted([row for row in audit
                       if row["point"] == key and row["angle_step_deg"] == 0.5],
                      key=lambda row: row["circle_sides"])
        axes[0].plot([row["circle_sides"] for row in data],
                     [row["worst_mec_radius_m"] for row in data], marker="o",
                     color=colors[key], label=labels[key])
    axes[0].set_xlabel("Circumscribed polygon sides N")
    axes[0].set_ylabel("Worst MEC radius (m)")
    axes[0].set_title("MEC convergence")
    axes[0].grid(alpha=0.2)
    axes[0].legend(frameon=False, fontsize=8)

    outer = sorted({(row["circle_sides"], row["arena_radial_excess_m"])
                    for row in audit})
    axes[1].loglog([row[0] for row in outer], [row[1] for row in outer],
                   marker="o", color="#b07aa1", linewidth=2)
    for sides, error in outer:
        axes[1].annotate(f"{error:.3f}", (sides, error), xytext=(0, 7),
                         textcoords="offset points", ha="center", fontsize=8)
    axes[1].set_xlabel("Circumscribed polygon sides N")
    axes[1].set_ylabel("Maximum radial excess (m)")
    axes[1].set_title("Conservative circle error")
    axes[1].grid(alpha=0.2, which="both")

    safe = payload["safe_margin_audit"]
    for name, color in (("recovered_outward", "#3b6ea8"),
                        ("near_boundary", "#c43c39")):
        data = [row for row in safe if row["point"] == name]
        axes[2].plot([row["angle_step_deg"] for row in data],
                     [row["margin_m"] for row in data], marker="o", color=color,
                     label=name.replace("_", " "))
    axes[2].set_xscale("log")
    axes[2].invert_xaxis()
    axes[2].set_xlabel("Safe-domain angular step (degree)")
    axes[2].set_ylabel("Reception margin (m)")
    axes[2].set_title("Safe-domain stability")
    axes[2].grid(alpha=0.2, which="both")
    axes[2].legend(frameon=False, fontsize=8)
    save(fig, output)


def _draw_geometry(axis: plt.Axes, first: np.ndarray, posterior: np.ndarray,
                   station: np.ndarray, q: np.ndarray, title: str,
                   show_labels: bool = True) -> None:
    center, radius = polygon_min_enclosing_circle(posterior)
    axis.fill(first[:, 0], first[:, 1], color="#bab0ac", alpha=0.25,
              label="First feasible set" if show_labels else None)
    axis.plot(first[:, 0], first[:, 1], color="#777777", linewidth=1)
    axis.fill(posterior[:, 0], posterior[:, 1], color="#4e79a7", alpha=0.48,
              label="Worst posterior" if show_labels else None)
    axis.plot(posterior[:, 0], posterior[:, 1], color="#2f5f8f", linewidth=1.2)
    axis.add_patch(plt.Circle(center, radius, fill=False, color="#e15759",
                              linewidth=1.7,
                              label="Minimum enclosing circle" if show_labels else None))
    axis.plot(station[0], station[1], "ko", ms=4,
              label="First monitor" if show_labels else None)
    axis.plot(q[0], q[1], marker="*", color="#f28e2b", ms=12,
              label="Second monitor" if show_labels else None)
    axis.plot(center[0], center[1], "+", color="#e15759", ms=8, mew=1.4)
    axis.set_aspect("equal", adjustable="datalim")
    axis.grid(alpha=0.15)
    axis.set_title(f"{title}\nworst MEC radius = {radius:.2f} m")
    axis.set_xlabel("x (m)")


def plot_worst_geometry(summary: dict, output: Path) -> None:
    representatives = summary["representative"]
    cases = [
        ("center", np.array([0.0, 0.0]), 0.0),
        ("outward edge", np.array([1700.0, 0.0]), 0.0),
        ("near tangent", np.array([1700.0, 0.0]), math.pi / 2.0),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 5.3), constrained_layout=True)
    for index, (name, station, bearing) in enumerate(cases):
        q = np.asarray(representatives[name]["point_m"], float)
        first = initial_region(station, bearing, 96)
        score = robust_score(first, q, 2.0, 96)
        base = clip_disk(first, q, 1500.0, 96)
        posterior = clip_wedge(base, q, score.worst_bearing_rad)
        axis = axes[index]
        _draw_geometry(axis, first, posterior, station, q, name)
        if index == 0:
            axis.set_ylabel("y (m)")
            axis.legend(frameon=False, fontsize=7, loc="upper left")

        zoom = inset_axes(axis, width="42%", height="42%", loc="lower right",
                         borderpad=1.0)
        _draw_geometry(zoom, first, posterior, station, q, "", show_labels=False)
        all_x, all_y = posterior[:, 0], posterior[:, 1]
        center, radius = polygon_min_enclosing_circle(posterior)
        pad = max(15.0, radius * 0.7)
        zoom.set_xlim(float(all_x.min() - pad), float(all_x.max() + pad))
        zoom.set_ylim(float(all_y.min() - pad), float(all_y.max() + pad))
        zoom.set_title("posterior zoom", fontsize=8)
        zoom.set_xlabel("")
        zoom.set_ylabel("")
        zoom.tick_params(labelsize=7)
        mark_inset(axis, zoom, loc1=2, loc2=4, fc="none", ec="0.55", lw=0.7)
    save(fig, output)


def plot_candidate_counts(rows: list[dict], output: Path) -> None:
    names = ["center", "outward edge", "near tangent"]
    totals, near_optimal, threshold = [], [], []
    for name in names:
        part = [row for row in rows if row["scenario"] == name]
        totals.append(len(part))
        near_optimal.append(sum(int(row["within_5pct"]) for row in part))
        threshold.append(sum(int(row["robust_le20"]) for row in part))
    x = np.arange(len(names))
    width = 0.24
    fig, axis = plt.subplots(figsize=(9.1, 5.0), constrained_layout=True)
    series = [(totals, "Reliable candidates", "#76b7b2"),
              (near_optimal, "Within 5% of optimum", "#e15759"),
              (threshold, "Robust MEC ≤ 20 m", "#4e79a7")]
    for index, (values, label, color) in enumerate(series):
        bars = axis.bar(x + (index - 1) * width, values, width,
                        label=label, color=color)
        axis.bar_label(bars, padding=3, fontsize=9)
    axis.set_xticks(x, names)
    axis.set_ylabel("Number of discrete candidate points")
    axis.set_title("Candidate-set sizes under two reporting rules")
    axis.grid(axis="y", alpha=0.2)
    axis.legend(frameon=False, ncols=3, loc="upper left")
    axis.set_ylim(0, max(totals) * 1.18)
    save(fig, output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path,
                        default=Path(__file__).parent / "results_mec")
    args = parser.parse_args()
    args.results.mkdir(parents=True, exist_ok=True)
    trials = read_csv(args.results / "trial_results.csv")
    candidates = read_csv(args.results / "candidate_points.csv")
    summary = json.loads((args.results / "summary.json").read_text(encoding="utf-8"))
    safe = json.loads((args.results / "safe_region_audit.json").read_text(encoding="utf-8"))
    convergence = json.loads((args.results / "convergence.json").read_text(encoding="utf-8"))

    plot_distributions(trials, args.results / "mec_distributions.png")
    plot_paired(trials, args.results / "paired_improvements.png")
    plot_safe_area(safe, args.results / "safe_area_bars.png")
    plot_convergence(convergence, args.results / "numerical_convergence.png")
    plot_worst_geometry(summary, args.results / "worst_case_geometry.png")
    plot_candidate_counts(candidates, args.results / "candidate_set_counts.png")
    print("generated 6 figures in", args.results)


if __name__ == "__main__":
    main()

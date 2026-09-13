"""Boundary-aware versus legacy full-sector reception-domain audit."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .mechanistic import (
    ANGLE_ERROR, RECEIVE_MIN, reception_supports, wrap_angle,
)


def legacy_mask(station: np.ndarray, bearing: float, points: np.ndarray) -> np.ndarray:
    delta = points - station
    radius = np.linalg.norm(delta, axis=1)
    beta = np.abs(wrap_angle(np.arctan2(delta[:, 1], delta[:, 0]) - bearing))
    return ((radius <= RECEIVE_MIN)
            & (beta + ANGLE_ERROR < math.pi / 2.0)
            & (radius <= 2.0 * RECEIVE_MIN * np.cos(beta + ANGLE_ERROR)))


def boundary_mask(station: np.ndarray, bearing: float, points: np.ndarray,
                  angle_step_deg: float, guard_m: float) -> tuple[np.ndarray, np.ndarray]:
    supports = reception_supports(station, bearing, angle_step_deg)
    margin = np.empty(len(points))
    block = 2048
    for start in range(0, len(points), block):
        part = points[start:start + block]
        distance = np.linalg.norm(
            part[:, None, :] - supports.points[None, :, :], axis=2)
        margin[start:start + block] = np.min(
            supports.radii[None, :] - distance, axis=1)
    return margin >= guard_m - 1e-7, margin


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid-step-m", type=float, default=10.0)
    parser.add_argument("--angle-step-deg", type=float, default=0.025)
    parser.add_argument("--guard-m", type=float, default=0.0)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).parent / "results_mec")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    cases = [
        ("center", np.array([0.0, 0.0]), 0.0),
        ("outward edge", np.array([1700.0, 0.0]), 0.0),
        ("near tangent", np.array([1700.0, 0.0]), math.pi / 2.0),
        ("inward edge", np.array([1700.0, 0.0]), math.pi),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.0), constrained_layout=True)
    rows = []
    for name, station, bearing in cases:
        offsets = np.arange(-1050.0, 1050.0 + args.grid_step_m / 2.0,
                            args.grid_step_m)
        xx, yy = np.meshgrid(station[0] + offsets, station[1] + offsets)
        points = np.column_stack((xx.ravel(), yy.ravel()))
        complete, margin = boundary_mask(
            station, bearing, points, args.angle_step_deg, args.guard_m)
        legacy = legacy_mask(station, bearing, points)
        cell_area = args.grid_step_m**2
        complete_area = float(np.sum(complete) * cell_area)
        legacy_area = float(np.sum(legacy) * cell_area)
        recovered = complete & ~legacy
        rows.append(dict(
            scenario=name,
            complete_area_km2=complete_area / 1e6,
            legacy_area_km2=legacy_area / 1e6,
            recovered_area_km2=float(np.sum(recovered) * cell_area / 1e6),
            legacy_fraction_percent=100.0 * legacy_area / complete_area,
            min_recovered_margin_m=(float(np.min(margin[recovered]))
                                    if np.any(recovered) else None),
        ))
        if name == "inward edge":
            continue
        axis = axes[len([row for row in rows[:-1] if row["scenario"] != "inward edge"])]
        image = np.zeros(len(points), dtype=int)
        image[complete] = 1
        image[legacy] = 2
        axis.contourf(xx, yy, image.reshape(xx.shape),
                      levels=[-0.5, 0.5, 1.5, 2.5],
                      colors=["#f5f5f5", "#f28e8e", "#56b4a6"])
        axis.plot(station[0], station[1], "ko", ms=4)
        arena = plt.Circle((0, 0), 1800.0, fill=False, color="0.45",
                           linestyle="--", linewidth=0.9)
        axis.add_patch(arena)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlim(station[0] - 1050, station[0] + 1050)
        axis.set_ylim(station[1] - 1050, station[1] + 1050)
        axis.set_title(name)
        axis.set_xlabel("x (m)")
    axes[0].set_ylabel("y (m)")
    from matplotlib.patches import Patch
    axes[0].legend(handles=[
        Patch(facecolor="#56b4a6", label="legacy safe"),
        Patch(facecolor="#f28e8e", label="recovered by boundary-aware set"),
    ], loc="upper left", fontsize=8)
    fig.savefig(args.output / "safe_region_comparison.png", dpi=180)
    plt.close(fig)

    payload = dict(config=dict(grid_step_m=args.grid_step_m,
                               angle_step_deg=args.angle_step_deg,
                               guard_m=args.guard_m), cases=rows)
    (args.output / "safe_region_audit.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

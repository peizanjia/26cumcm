"""Numerical-resolution audit for the representative full-sector case."""

import argparse
import json
from pathlib import Path

import numpy as np

from .mechanistic import (
    boundary_aware_safe, disk_polygon_radial_excess, initial_region,
    reception_margin_from_supports, reception_supports, robust_score,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).parent / "results_mec" / "convergence.json")
    args = parser.parse_args()
    points = {
        "center_minimax": np.array([838.670567940354, -544.6390350128164]),
        "edge_outward": np.array([1776.95768441, 98.5013442]),
        "analytic_far_range": np.array([12000/13, 5000/13]),
    }
    rows = []
    for sides in (64, 128, 256, 512):
        first = initial_region(np.zeros(2), 0.0, sides)
        edge_first = initial_region(np.array([1700.0, 0.0]), 0.0, sides)
        for step in (2.0, 1.0, 0.5):
            for name, point in points.items():
                region = edge_first if name == "edge_outward" else first
                score = robust_score(region, point, step, sides)
                rows.append(dict(circle_sides=sides, angle_step_deg=step,
                                 point=name,
                                 worst_mec_radius_m=score.worst_mec_radius_m,
                                 worst_diameter_m=score.worst_diameter_m,
                                 worst_area_m2=score.worst_area_m2,
                                 arena_radial_excess_m=disk_polygon_radial_excess(1800.0, sides)))
    safe_rows = []
    station = np.array([1700.0, 0.0])
    safe_points = {
        "recovered_outward": np.array([1800.0, 800.0]),
        "near_boundary": np.array([1700.0, 990.0]),
    }
    for step in (0.2, 0.1, 0.05, 0.025, 0.0125):
        supports = reception_supports(station, 0.0, step)
        for name, point in safe_points.items():
            safe_rows.append(dict(
                angle_step_deg=step, point=name,
                margin_m=reception_margin_from_supports(supports, point),
                safe=boundary_aware_safe(station, 0.0, point,
                                         step, 0.0, supports)))
    payload = {
        "audit": rows,
        "safe_margin_audit": safe_rows,
        "max_center_mec_spread_m": float(np.ptp(
            [r["worst_mec_radius_m"] for r in rows
             if r["point"] == "center_minimax"])),
        "max_edge_mec_spread_m": float(np.ptp(
            [r["worst_mec_radius_m"] for r in rows
             if r["point"] == "edge_outward"])),
        "interpretation": "MEC radius is audited over circle polygon and future-bearing resolutions; state-aware safe margins are audited separately down to 0.0125 degree."
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

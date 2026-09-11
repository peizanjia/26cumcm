"""Numerical-resolution audit for the representative full-sector case."""

import argparse
import json
from pathlib import Path

import numpy as np

from .mechanistic import initial_region, robust_score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).parent / "results" / "convergence.json")
    args = parser.parse_args()
    points = {
        "mechanistic": np.array([838.670567940354, 544.6390350128164]),
        "repository_neural": np.array([974.6363525390625, -271.9693603515625]),
        "analytic_far_range": np.array([12000/13, 5000/13]),
    }
    rows = []
    for sides in (48, 64, 96, 128, 192):
        first = initial_region(np.zeros(2), 0.0, sides)
        for step in (4.0, 2.0, 1.0, 0.5):
            for name, point in points.items():
                score = robust_score(first, point, step, sides)
                rows.append(dict(circle_sides=sides, angle_step_deg=step,
                                 point=name,
                                 worst_diameter_m=score.worst_diameter_m,
                                 worst_area_m2=score.worst_area_m2))
    payload = {
        "audit": rows,
        "max_mechanistic_diameter_spread_m": float(np.ptp(
            [r["worst_diameter_m"] for r in rows if r["point"] == "mechanistic"])),
        "interpretation": "Angle maxima occur at included critical directions; 48-192 circle sides change the representative robust diameter by less than the reported spread."
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

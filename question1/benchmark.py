"""Measure full geometry calls on reproducible bounded synthetic cases."""

import argparse
import time

import numpy as np

from .geometry import intersect_bearings
from .simulate import BearingEnvironment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=200)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    print("points,median_ms,p95_ms,vertices")
    for count in (2, 4, 10, 50, 100):
        env = BearingEnvironment(seed=42)
        points = env.sample_monitoring_points(count)
        angles = [env.measure(p)["svd_deg"] for p in points]
        result = intersect_bearings(points, angles)
        if result.status != "bounded":
            raise AssertionError("Benchmark expects bounded cases")
        times = []
        for _ in range(args.repeats):
            start = time.perf_counter()
            intersect_bearings(points, angles)
            times.append((time.perf_counter() - start) * 1000)
        print(f"{count},{np.median(times):.4f},{np.percentile(times,95):.4f},{len(result.vertices)}")


if __name__ == "__main__":
    main()

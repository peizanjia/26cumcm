"""Reproducible single-source simulation; run: python -m question1.simulate."""

import argparse
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import struct

import numpy as np

from .geometry import bearing_halfplanes, intersect_bearings, polygon_area, polygon_diameter


def sample_disk(rng: np.random.Generator, count: int, radius: float) -> np.ndarray:
    """Uniform in area: r=R*sqrt(U), not R*U."""
    if count < 0 or not np.isfinite(radius) or radius <= 0:
        raise ValueError("count must be nonnegative; radius must be positive and finite")
    angle = rng.uniform(0, 2 * np.pi, count)
    r = radius * np.sqrt(rng.random(count))
    return np.column_stack((r * np.cos(angle), r * np.sin(angle)))


@dataclass
class BearingEnvironment:
    """One omnidirectional source with a fixed seeded error at each coordinate.

    Errors at distinct coordinates are independent by modeling assumption.
    No claim is made that this reproduces the official simulator's error field.
    """

    seed: int = 42
    radius: float = 1800.0
    noise: str = "uniform"
    error_deg: float = 1.0
    sigma_deg: float = 1 / 3
    receive_radius: float | None = None
    source: np.ndarray = field(init=False)
    rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self):
        if not isinstance(self.seed, (int, np.integer)) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if not np.isfinite(self.radius) or self.radius <= 0:
            raise ValueError("radius must be positive and finite")
        if self.noise not in ("uniform", "truncnorm"):
            raise ValueError("noise must be uniform or truncnorm")
        if not np.isfinite(self.error_deg) or not 0 < self.error_deg < 90:
            raise ValueError("error_deg must be between 0 and 90")
        if not np.isfinite(self.sigma_deg) or self.sigma_deg <= 0:
            raise ValueError("sigma_deg must be positive and finite")
        self.rng = np.random.default_rng(self.seed)
        self.source = sample_disk(self.rng, 1, self.radius)[0]
        if self.receive_radius is None:
            self.receive_radius = float(self.rng.uniform(1000, 1500))
        if not np.isfinite(self.receive_radius) or not 1000 <= self.receive_radius <= 1500:
            raise ValueError("receive_radius must be within [1000,1500] metres")

    def error_at(self, point) -> float:
        p = np.asarray(point, dtype=float)
        if p.shape != (2,) or not np.isfinite(p).all():
            raise ValueError("point must be a finite coordinate pair")
        # Hashing exact coordinates makes the field independent of query order.
        # Normalize signed zero. Nearby coordinates need not have nearby errors.
        x, y = (float(v) if v != 0 else 0.0 for v in p)
        key = str(self.seed).encode("ascii") + b":" + struct.pack("!dd", x, y)
        point_seed = int.from_bytes(hashlib.blake2b(key, digest_size=16).digest(), "big")
        rng = np.random.default_rng(point_seed)
        if self.noise == "uniform":
            return float(rng.uniform(-self.error_deg, self.error_deg))
        from scipy.stats import truncnorm
        return float(truncnorm.rvs(-self.error_deg / self.sigma_deg,
                                  self.error_deg / self.sigma_deg,
                                  scale=self.sigma_deg, random_state=rng))

    def measure(self, point) -> dict:
        p = np.asarray(point, dtype=float)
        if p.shape != (2,) or not np.isfinite(p).all():
            raise ValueError("point must be a finite coordinate pair")
        delta = self.source - p
        distance = float(np.linalg.norm(delta))
        if distance > self.receive_radius:
            return {"measure_result": "no_signal"}
        if distance <= 5:
            return {"measure_result": "near"}
        true = float(np.degrees(np.arctan2(delta[1], delta[0])) % 360)
        error = self.error_at(p)
        return {"measure_result": "direction", "svd_deg": (true + error) % 360,
                "true_deg": true, "error_deg": error, "distance_m": distance}

    def sample_monitoring_points(self, count: int, min_distance: float = 50.0) -> np.ndarray:
        """Uniform in the valid arena/receiving-annulus intersection by rejection.

        Uses ground truth to construct an observable test case, not a search
        policy a robot can execute with an unknown source.
        """
        if not isinstance(count, (int, np.integer)) or count < 1:
            raise ValueError("count must be a positive integer")
        if not np.isfinite(min_distance) or not 5 < min_distance < self.receive_radius:
            raise ValueError("min_distance must be >5 and <receive_radius")
        selected = []
        for _ in range(10000):
            candidates = sample_disk(self.rng, max(64, count - len(selected)), self.radius)
            distances = np.linalg.norm(candidates - self.source, axis=1)
            selected.extend(candidates[(distances >= min_distance) &
                                       (distances <= self.receive_radius)])
            if len(selected) >= count:
                return np.array(selected[:count])
        raise RuntimeError("Cannot sample enough visible monitoring points; check geometry")


def plot_scene(env, points, bearings, result, output, show=False):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Polygon

    colors = plt.get_cmap("tab10")
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6), layout="constrained")
    poly = result.vertices
    diameter, endpoints = polygon_diameter(poly)
    for ax in axes:
        ax.set_aspect("equal")
        ax.set_xlabel("East x (m)")
        ax.set_ylabel("North y (m)")
        ax.grid(alpha=0.18)
        ax.add_patch(Circle((0, 0), env.radius, fill=False, ec="#94a3b8", lw=1.2))
        for i, (p, theta) in enumerate(zip(points, bearings)):
            color = colors(i % 10)
            length = max(5 * env.radius, float(np.linalg.norm(env.source - p)) * 3,
                         float(np.max(np.linalg.norm(poly - p, axis=1))) * 1.1 if len(poly) else 0)
            angles = np.deg2rad([theta - env.error_deg, theta + env.error_deg])
            tips = p + length * np.column_stack((np.cos(angles), np.sin(angles)))
            ax.add_patch(Polygon(np.vstack((p, tips)), color=color, alpha=0.09))
            for tip in tips:
                ax.plot([p[0], tip[0]], [p[1], tip[1]], color=color, lw=0.8, alpha=0.7)
            mid = p + length * np.array([np.cos(np.deg2rad(theta)), np.sin(np.deg2rad(theta))])
            ax.plot([p[0], mid[0]], [p[1], mid[1]], "--", color=color, lw=0.9)
        if len(poly) >= 3:
            ax.add_patch(Polygon(poly, fc="#14b8a6", ec="#0f766e", lw=2, alpha=0.5,
                                 label="Bearing intersection"))
        elif len(poly):
            ax.plot(poly[:, 0], poly[:, 1], "o-", color="#0f766e", label="Degenerate intersection")
        ax.scatter(*env.source, marker="*", s=180, color="#dc2626", edgecolor="white",
                   linewidth=0.7, zorder=10, label="True source")
    ax = axes[0]
    ax.scatter(points[:, 0], points[:, 1], s=48,
               c=[colors(i % 10) for i in range(len(points))], edgecolors="white", zorder=8,
               label="Monitoring points")
    for i, p in enumerate(points):
        ax.annotate(f"S{i+1}", p, xytext=(5, 7), textcoords="offset points", fontsize=9)
    ax.set(xlim=(-1.12 * env.radius, 1.12 * env.radius),
           ylim=(-1.12 * env.radius, 1.12 * env.radius), title="Arena and noisy bearing wedges")
    ax.legend(loc="upper right", fontsize=9)
    ax = axes[1]
    if len(poly):
        low, high = poly.min(axis=0), poly.max(axis=0)
        center = (low + high) / 2
        half = max(float(np.max(high - low)) * 0.75, 5.0)
        ax.set(xlim=(center[0] - half, center[0] + half),
               ylim=(center[1] - half, center[1] + half))
        ax.plot(endpoints[:, 0], endpoints[:, 1], "o--", color="#0f172a", lw=1.2,
                markersize=4, label=f"Diameter: {diameter:.2f} m")
        ax.set_title(f"Full intersection | area {polygon_area(poly):.2f} m²")
        ax.ticklabel_format(useOffset=False, style="plain")
        ax.legend(loc="best", fontsize=9)
    else:
        ax.set(xlim=(env.source[0]-100, env.source[0]+100),
               ylim=(env.source[1]-100, env.source[1]+100),
               title=f"Intersection is {result.status}; no finite polygon")
    fig.suptitle(f"Question 1 · seed {env.seed} · {len(points)} stations · "
                 f"{env.noise} noise ±{env.error_deg:g}°", fontsize=15, fontweight="bold")
    fig.savefig(output / "localization.png", dpi=180)
    fig.savefig(output / "localization.svg")
    if show:
        plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--points", type=int, default=4)
    parser.add_argument("--radius", type=float, default=1800)
    parser.add_argument("--noise", choices=["uniform", "truncnorm"], default="uniform")
    parser.add_argument("--sigma", type=float, default=1 / 3, help="truncated Gaussian sigma in degrees")
    parser.add_argument("--error-deg", type=float, default=1)
    parser.add_argument("--receive-radius", type=float, default=None)
    parser.add_argument("--min-distance", type=float, default=50)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "output")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    try:
        env = BearingEnvironment(seed=args.seed, radius=args.radius, noise=args.noise,
                                 error_deg=args.error_deg, sigma_deg=args.sigma,
                                 receive_radius=args.receive_radius)
        points = env.sample_monitoring_points(args.points, args.min_distance)
        measurements = [env.measure(p) for p in points]
        bearings = np.array([m["svd_deg"] for m in measurements])
        result = intersect_bearings(points, bearings, env.error_deg)
    except ValueError as exc:
        parser.error(str(exc))
    a, b = bearing_halfplanes(points, bearings, env.error_deg)
    truth_feasible = bool(np.all(a @ env.source <= b + 1e-7))
    if result.status == "empty" or not truth_feasible:
        raise AssertionError("Bounded-error observations must retain the true source")
    diameter, endpoints = polygon_diameter(result.vertices)
    report = {"seed": args.seed, "arena_radius_m": env.radius,
              "noise": env.noise, "error_bound_deg": env.error_deg,
              "sigma_deg": env.sigma_deg if env.noise == "truncnorm" else None,
              "receive_radius_m": env.receive_radius, "min_distance_m": args.min_distance,
              "source": env.source.tolist(), "monitoring_points": points.tolist(),
              "measurements": measurements, "intersection_status": result.status,
              "vertices": result.vertices.tolist(), "true_source_feasible": truth_feasible,
              "area_m2": polygon_area(result.vertices) if result.status == "bounded" else None,
              "diameter_m": diameter if result.status == "bounded" else None,
              "diameter_endpoints": endpoints.tolist(),
              "arena_clipped": False,
              "assumptions": ["Independent fixed errors at distinct exact coordinates",
                              "Uniform-area source and valid monitoring point sampling",
                              "Synthetic environment, not the official simulator"]}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "simulation.json").write_text(json.dumps(report, indent=2, ensure_ascii=False,
                                                           allow_nan=False), encoding="utf-8")
    data = np.column_stack((points, bearings, [m["true_deg"] for m in measurements],
                            [m["error_deg"] for m in measurements]))
    np.savetxt(args.output / "measurements.csv", data, delimiter=",", fmt="%.12f",
               header="x_m,y_m,bearing_deg,true_bearing_deg,error_deg", comments="")
    plot_scene(env, points, bearings, result, args.output, args.show)
    print(f"status={result.status}; vertices={len(result.vertices)}; truth_feasible={truth_feasible}")
    if result.status == "bounded":
        print(f"area={report['area_m2']:.6f} m^2; diameter={diameter:.6f} m")
    print(f"Outputs: {args.output.resolve()}")


if __name__ == "__main__":
    main()

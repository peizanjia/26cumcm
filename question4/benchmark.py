"""Local directional-emitter route benchmark (never connects to the official server).

This is a route-level benchmark, not the final Q4 planner.  Every scene has 10--16
sources, some omni and some 180-degree emitters.  At each stop all 20 channels are
scanned, so the comparison isolates the geometry of candidate routes.  A source is
counted as discovered after a ``direction`` or ``near`` response; it is not cleared.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path

R_DOMAIN = 1800.0
R_MIN, R_MAX = 1000.0, 1500.0
N_CHANNELS = 20


@dataclass(frozen=True)
class Source:
    channel: int
    x: float
    y: float
    radius: float
    directional: bool
    orientation: float


def make_scene(seed: int) -> list[Source]:
    rng = random.Random(seed)
    channels = rng.sample(range(1, N_CHANNELS + 1), rng.randint(10, 16))
    out: list[Source] = []
    for channel in channels:
        r = R_DOMAIN * math.sqrt(rng.random())
        a = rng.random() * 2 * math.pi
        out.append(Source(channel, r * math.cos(a), r * math.sin(a),
                          rng.uniform(R_MIN, R_MAX), rng.random() < 0.55,
                          rng.random() * 2 * math.pi))
    return out


def illuminated(source: Source, p: tuple[float, float]) -> bool:
    if not source.directional:
        return True
    dx, dy = p[0] - source.x, p[1] - source.y
    return dx * math.cos(source.orientation) + dy * math.sin(source.orientation) >= -1e-9


def measure(source: Source, p: tuple[float, float]) -> str:
    d = math.hypot(p[0] - source.x, p[1] - source.y)
    if d <= 5.0 and illuminated(source, p):
        return "near"
    if d <= source.radius and illuminated(source, p):
        return "direction"
    return "no_signal"


def ring(radius: float, n: int) -> list[tuple[float, float]]:
    return [(radius * math.cos(2 * math.pi * i / n),
             radius * math.sin(2 * math.pi * i / n)) for i in range(n)]


def routes() -> dict[str, list[tuple[float, float]]]:
    # All routes begin with the mandatory origin scan.  Values outside 1800 are
    # intentional: outward-facing boundary sources cannot be covered internally.
    origin = [(0.0, 0.0)]
    return {
        "single_ring": origin + ring(1800.0, 36),
        "two_layer": origin + ring(900.0, 24) + ring(1950.0, 32),
        "spiral": origin + [
            (r * math.cos(t), r * math.sin(t))
            for i in range(120)
            for t, r in [(i * math.pi / 9, 80.0 + i * (2120.0 - 80.0) / 119)]
        ],
        # A geometry-only farthest-point route.  It is a proxy for the proposed
        # direction-gap candidate generator and uses no scene truth.
        "gap_candidates": origin + _farthest_candidates(),
    }


def _farthest_candidates() -> list[tuple[float, float]]:
    pool: list[tuple[float, float]] = []
    for r, n in ((700.0, 16), (1200.0, 24), (1750.0, 32), (2050.0, 32)):
        pool.extend(ring(r, n))
    pool.extend([(x, y) for x in (-1000.0, 0.0, 1000.0) for y in (-1000.0, 0.0, 1000.0)])
    selected: list[tuple[float, float]] = []
    while pool and len(selected) < 70:
        if not selected:
            k = max(range(len(pool)), key=lambda i: math.hypot(*pool[i]))
        else:
            k = max(range(len(pool)), key=lambda i: min(math.dist(pool[i], q) for q in selected))
        selected.append(pool.pop(k))
    return selected


def evaluate(seed: int, route: list[tuple[float, float]]) -> dict:
    scene = make_scene(seed)
    found: set[int] = set()
    first: dict[int, float] = {}
    elapsed = 0.0
    previous = route[0]
    current_channel = 1  # detector starts on channel 1 at the origin
    for p in route:
        elapsed += math.dist(previous, p) / 5.0
        for channel in range(1, N_CHANNELS + 1):
            elapsed += 5.0 + (channel != current_channel)
            current_channel = channel
            source = next((s for s in scene if s.channel == channel), None)
            if source is not None and channel not in found and measure(source, p) in ("near", "direction"):
                found.add(channel)
                first[channel] = elapsed
        previous = p
    return {
        "seed": seed,
        "source_count": len(scene),
        "directional_count": sum(s.directional for s in scene),
        "found": len(found),
        "found_fraction": len(found) / len(scene),
        "complete": len(found) == len(scene),
        "time_s": elapsed,
        "first_discovery_s": statistics.mean(first.values()) if first else None,
        "last_discovery_s": max(first.values()) if first else None,
        "movement_s": sum(math.dist(route[i - 1], route[i]) for i in range(1, len(route))) / 5.0,
        "stops": len(route),
        "truth": [asdict(s) for s in scene],
    }


def summarize(rows: dict[str, list[dict]]) -> dict:
    out = {}
    for name, values in rows.items():
        out[name] = {
            "scenes": len(values),
            "complete_rate": statistics.mean(r["complete"] for r in values),
            "mean_found_fraction": statistics.mean(r["found_fraction"] for r in values),
            "mean_time_s": statistics.mean(r["time_s"] for r in values),
            "mean_movement_s": statistics.mean(r["movement_s"] for r in values),
            "mean_first_discovery_s": statistics.mean(r["first_discovery_s"] for r in values if r["first_discovery_s"] is not None),
            "mean_last_discovery_s": statistics.mean(r["last_discovery_s"] for r in values if r["last_discovery_s"] is not None),
            "mean_stops": statistics.mean(r["stops"] for r in values),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=100)
    ap.add_argument("--seed-start", type=int, default=20274000)
    ap.add_argument("--output", default="question4/outputs/route_benchmark.json")
    args = ap.parse_args()
    route_map = routes()
    rows = {name: [evaluate(args.seed_start + i, route) for i in range(args.count)]
            for name, route in route_map.items()}
    payload = {
        "kind": "q4_local_directional_route_benchmark",
        "evaluation": "synthetic_only_route_discovery_not_official_score",
        "seed_start": args.seed_start,
        "count": args.count,
        "physics": {"domain_radius_m": R_DOMAIN, "receive_radius_m": [R_MIN, R_MAX],
                    "directional_beam_deg": 180, "movement_speed_mps": 5,
                    "measure_s": 5, "switch_s": 1},
        "route_stops": {k: len(v) for k, v in route_map.items()},
        "summary": summarize(rows),
        "cases": rows,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

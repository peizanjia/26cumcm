"""Paired Q4 search-route study with observable geometric stopping.

Search time excludes bearing localization and optical clearing. Source truth is
used only by the sensor and evaluator. The policy scans channels 1..20 initially,
then unresolved channels; it stops on a geometric domain certificate or 16 hits.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy.spatial import ConvexHull, Delaunay, QhullError

from question4.benchmark import Source, make_scene
from .routes import baseline_routes, candidate_routes, scaffold_points, zigzag

BASE = Path(__file__).resolve().parent
DOMAIN = 1800.0
RECEIVE_MIN = 1000.0


def make_cells(side=175.0):
    cells = []
    axis = np.arange(-2100.0, 2100.0, side)
    for x in axis:
        for y in axis:
            closest = np.clip([0.0, 0.0], [x, y], [x + side, y + side])
            if np.dot(closest, closest) <= DOMAIN**2:
                cells.append([[x, y], [x + side, y], [x + side, y + side], [x, y + side]])
    return np.asarray(cells)


def certify_cells(points, cells):
    """Sufficient whole-cell local hull test; never a mere center test."""
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    certified = np.zeros(len(cells), dtype=bool)
    for k, cell in enumerate(cells):
        distances = np.linalg.norm(points[:, None, :] - cell[None, :, :], axis=2)
        nearby = points[np.max(distances, axis=1) <= RECEIVE_MIN - 1e-7]
        if len(nearby) < 3:
            continue
        try:
            hull = ConvexHull(nearby)
            certified[k] = np.all(cell @ hull.equations[:, :2].T + hull.equations[:, 2] <= 0)
        except QhullError:
            pass
    return certified


def triangle_min_radius(triangles):
    """Distance from origin to each closed triangle, including its interior."""
    a = triangles
    b = np.roll(a, -1, axis=1)
    v = b - a
    t = np.clip(-np.sum(a * v, axis=2) / np.maximum(np.sum(v * v, axis=2), 1e-30), 0, 1)
    distances = np.linalg.norm(a + t[:, :, None] * v, axis=2).min(axis=1)
    cross = v[:, :, 0] * (-a[:, :, 1]) - v[:, :, 1] * (-a[:, :, 0])
    inside = np.all(cross >= 0, axis=1) | np.all(cross <= 0, axis=1)
    distances[inside] = 0
    return distances


def certificate_details(points):
    """Continuous sufficient certificate: a short-edge triangulation covers D.

    A triangle's diameter bounds the distance from every interior point to each
    vertex. A source's closed emitting halfplane must contain at least one of
    the three vertices. Strict numerical margins make the check conservative.
    """
    p = np.unique(np.asarray(points, dtype=float).reshape(-1, 2), axis=0)
    empty = dict(certified=False, triangles=[], hull_inradius_m=0.0, max_relevant_edge_m=None)
    if len(p) < 3:
        return empty
    try:
        hull = ConvexHull(p)
        tris = p[Delaunay(p).simplices]
    except QhullError:
        return empty
    inradius = float(np.min(-hull.equations[:, 2]))
    edges = np.linalg.norm(tris - np.roll(tris, -1, axis=1), axis=2).max(axis=1)
    relevant = triangle_min_radius(tris) <= DOMAIN + 1e-7
    certified = inradius >= DOMAIN + 1e-7 and bool(np.all(edges[relevant] <= RECEIVE_MIN - 1e-7))
    result = dict(certified=certified, triangles=tris[edges <= RECEIVE_MIN - 1e-7].tolist(),
                hull_inradius_m=inradius,
                max_relevant_edge_m=float(edges[relevant].max()) if relevant.any() else None)
    if not certified:
        # A new point can change the Delaunay mesh, but cannot invalidate a
        # previously sufficient subset. Retain the explicit proven repair mesh.
        subset = zigzag(990.0, 1900.0, 24)
        if len(p) > len(subset) and all(np.linalg.norm(p-q, axis=1).min() < 1e-7 for q in subset):
            proof = certificate_details(subset)
            result.update(proof, proof_points=subset.tolist(), proof_kind="certified_subset")
    return result


def certify_domain(points):
    return certificate_details(points)["certified"]


def prepare_route(row):
    """Append the shared 25-stop repair and stop at an observed certificate.

    Nothing here depends on a scene. The original source-free geometry alone
    decides repair points. At runtime an unresolved channel measures each stop.
    """
    requested = np.asarray(row["points"], dtype=float)
    if len(requested) == 0 or np.linalg.norm(requested[0]) > 1e-9:
        raise ValueError("Every route must begin with the origin")
    out = dict(row)
    out["base_points"] = requested.tolist()
    points = []
    certified = False
    base_used = 0
    for q in requested:
        points.append(q.tolist())
        base_used += 1
        if certify_domain(points):
            certified = True
            break
    # The newly proved 25-stop zigzag is a cheaper common repair than the old
    # 45-stop square scaffold. The latter remains a final numerical safeguard.
    repair_pool = np.vstack((zigzag(990.0, 1900.0, 24), scaffold_points()))
    primary_repair = zigzag(990.0, 1900.0, 24)
    repair = [np.asarray(q) for q in primary_repair
              if not any(np.linalg.norm(np.asarray(q) - p) < 1e-7 for p in points)]
    while not certified and repair:
        j = int(np.argmin([np.linalg.norm(q - points[-1]) for q in repair]))
        points.append(repair.pop(j).tolist())
        certified = certify_domain(points)
    if not certified:
        # A union of additional stops cannot invalidate an existing proof;
        # check the known subset if Delaunay's sufficient test changed mesh.
        certified = all(any(np.linalg.norm(q-np.asarray(p)) < 1e-7 for p in points)
                        for q in primary_repair)
    if not certified:
        for q in repair_pool:
            if not any(np.linalg.norm(q-np.asarray(p)) < 1e-7 for p in points):
                points.append(q.tolist())
            if certify_domain(points):
                certified = True
                break
    if not certified:
        raise RuntimeError(f"Finite scaffold failed certification: {row['id']}")
    out.update(points=points, base_used=base_used, repair_count=len(points) - base_used,
               certificate_complete=True, certificate_prefix=len(points), stops=len(points),
               path_length_m=float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum()))
    return out


def hit_matrix(scene, points):
    points = np.asarray(points, dtype=float)
    if not scene:
        return np.zeros((len(points), 0), dtype=bool)
    source_xy = np.array([[s.x, s.y] for s in scene])
    delta = points[:, None, :] - source_xy[None, :, :]
    radii = np.array([s.radius for s in scene])
    normal = np.array([[math.cos(s.orientation), math.sin(s.orientation)] for s in scene])
    directional = np.array([s.directional for s in scene])
    front = np.sum(delta * normal[None, :, :], axis=2) >= -1e-9
    return (np.sum(delta * delta, axis=2) <= radii[None, :]**2 + 1e-8) & (front | ~directional)


def simulate(scene, prepared, scan_mode="unresolved", keep_replay=False):
    """Run an ordered search plan. No true-source-count stopping condition."""
    if scan_mode not in ("unresolved", "all"):
        raise ValueError(scan_mode)
    points = np.asarray(prepared["points"], dtype=float)
    hits = hit_matrix(scene, points)
    source_columns = {s.channel: j for j, s in enumerate(scene)}
    found = set()
    first = {}
    movement = measure = switch = 0.0
    repairs = 0.0
    pos = np.zeros(2)
    current = 1
    stop_history = []
    measurements = 0
    executed_stops = 0
    certified = False
    for i, q in enumerate(points):
        old_time = movement + measure + switch
        movement += float(np.linalg.norm(q - pos)) / 5.0
        arrival = movement + measure + switch
        channels = []
        new = []
        # Origin always scans all twenty channels, even after 16 detections.
        for c in range(1, 21):
            if i > 0 and scan_mode == "unresolved" and c in found:
                continue
            switch += float(c != current)
            current = c
            measure += 5.0
            measurements += 1
            channels.append(c)
            j = source_columns.get(c)
            if j is not None and hits[i, j] and c not in found:
                found.add(c)
                new.append(c)
                first[str(c)] = movement + measure + switch
            if i > 0 and prepared.get("allow_sixteen_stop", True) and len(found) >= 16:
                break
        pos = q
        executed_stops += 1
        end = movement + measure + switch
        phase = "base" if i < prepared.get("base_used", len(points)) else "repair"
        if phase == "repair":
            repairs += end - old_time
        certified = bool(prepared.get("certificate_complete", False) and
                         i + 1 >= prepared.get("certificate_prefix", len(points)))
        if keep_replay:
            stop_history.append(dict(xy=q.tolist(), arrival_s=arrival, end_s=end,
                channels=channels, new_channels=new, phase=phase,
                certified_fraction=1.0 if certified else None))
        if certified or (prepared.get("allow_sixteen_stop", True) and len(found) >= 16):
            break
    elapsed = movement + measure + switch
    complete = len(found) == len(scene)  # Evaluator only; never used above to stop.
    n = len(scene)
    row = dict(time_s=elapsed, movement_s=movement, measure_s=measure, switch_s=switch,
        repair_s=repairs, complete=complete, found=len(found), source_count=n,
        missed_channels=sorted({s.channel for s in scene} - found),
        s_per_source=elapsed / n if n else None, measurements=measurements,
        executed_stops=executed_stops, first_discovery=first,
        all_found_s=max(first.values()) if complete and first else None,
        mean_source_first_s=float(np.mean(list(first.values()))) if first else None,
        observable_stop="geometry_certificate" if certified else "sixteen_found" if prepared.get("allow_sixteen_stop", True) and len(found) >= 16 else "plan_exhausted")
    if keep_replay:
        row["stops"] = stop_history
    return row


def raw_plan(row):
    points = row.get("base_points", row["points"])
    return dict(points=points, base_used=len(points), certificate_complete=False, allow_sixteen_stop=False)


def aggregate(rows):
    return dict(mean_s_per_source=float(np.mean([r["s_per_source"] for r in rows])),
        mean_time_s=float(np.mean([r["time_s"] for r in rows])),
        complete_rate=float(np.mean([r["complete"] for r in rows])),
        mean_movement_s=float(np.mean([r["movement_s"] for r in rows])),
        mean_measure_s=float(np.mean([r["measure_s"] for r in rows])),
        mean_switch_s=float(np.mean([r["switch_s"] for r in rows])),
        mean_movement_s_per_source=float(np.mean([r["movement_s"]/r["source_count"] for r in rows])),
        mean_measure_s_per_source=float(np.mean([r["measure_s"]/r["source_count"] for r in rows])),
        mean_switch_s_per_source=float(np.mean([r["switch_s"]/r["source_count"] for r in rows])),
        mean_path_m=float(np.mean([r["movement_s"]*5 for r in rows])),
        mean_stops=float(np.mean([r["executed_stops"] for r in rows])),
        mean_measurements=float(np.mean([r["measurements"] for r in rows])),
        mean_repair_s=float(np.mean([r["repair_s"] for r in rows])),
        mean_source_first_s=float(np.mean([r["mean_source_first_s"] for r in rows if r["mean_source_first_s"] is not None])),
        missed_total=sum(len(r["missed_channels"]) for r in rows),
        p90_s_per_source=float(np.quantile([r["s_per_source"] for r in rows], .9)),
        max_s_per_source=max(r["s_per_source"] for r in rows),
        total_sources=sum(r["source_count"] for r in rows))


def geometric_map(points, step=75.0):
    """At each displayed location, exact possible halfplane gap for R=1000.

    A nonzero cell is a concrete possible-miss location. A zero-valued sampled
    map is not a continuous certificate; certificate_details supplies that.
    """
    p = np.asarray(points, dtype=float)
    axis = np.arange(-1800.0, 1800.01, step)
    values = np.full((len(axis), len(axis)), -1.0)
    witness = None
    count = holes = 0
    for iy, y in enumerate(axis):
        for ix, x in enumerate(axis):
            if x*x + y*y > DOMAIN**2:
                continue
            count += 1
            g = np.array([x, y])
            d = p - g
            length = np.linalg.norm(d, axis=1)
            if np.any(length < 1e-9):
                values[iy, ix] = 0
                continue
            nearby = d[length <= RECEIVE_MIN + 1e-9]
            if not len(nearby):
                gap, orientation = 2 * np.pi, 0.0
                mass = 1.0
            else:
                angles = np.sort(np.mod(np.arctan2(nearby[:, 1], nearby[:, 0]), 2*np.pi))
                gaps = np.diff(np.r_[angles, angles[0] + 2*np.pi])
                j = int(np.argmax(gaps))
                gap = gaps[j]
                orientation = (angles[j] + gap/2) % (2*np.pi)
                mass = max(0.0, float((gap-np.pi)/(2*np.pi)))
            if gap <= np.pi + 1e-10:
                mass = 0.0
            values[iy, ix] = mass
            if mass > 0:
                holes += 1
                source = Source(1, x, y, 1000.0, True, float(orientation))
                # Do not publish a counterexample unless the independent sensor agrees.
                if not hit_matrix([source], p).any() and (witness is None or mass > witness["orientation_fraction"]):
                    witness = dict(x=float(x), y=float(y), orientation=float(orientation),
                                   radius=1000.0, orientation_fraction=mass)
    result = certificate_details(p)
    result.update(points=p.tolist(), grid=dict(xs=axis.tolist(), ys=axis.tolist(), values=values.tolist()),
                  witness=witness, sampled_positions=count, sampled_holes=holes,
                  sampled_covered_fraction=1-holes/max(count, 1))
    return result


def stress_scenes():
    out = []
    for offset in (0.0, 0.07, 0.173, 0.31):
        sources = []
        for k in range(12):
            a = offset + k * 2*np.pi/12
            # Half exactly on source boundary, half near origin, all face outward.
            r = (1800.0, 500.0, 925.0)[k % 3]
            sources.append(Source(k+1, r*math.cos(a), r*math.sin(a), 1000.0, True, a))
        out.append(sources)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev-count", type=int, default=64)
    ap.add_argument("--test-count", type=int, default=200)
    ap.add_argument("--output", default=str(BASE / "outputs"))
    args = ap.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "benchmark.json").exists():
        raise RuntimeError("Choose a fresh output directory; preserve previous results")
    started = time.perf_counter()
    dev_start, test_start = 20278000, 20279000
    if args.dev_count > test_start - dev_start:
        raise ValueError("Development and test seeds overlap")
    dev = [make_scene(dev_start+i) for i in range(args.dev_count)]
    bases = baseline_routes()
    candidates = candidate_routes()
    optimized, evaluated = {}, []
    prepared_candidates = {}
    for k, row in enumerate(candidates):
        p = prepare_route(row)
        stats = aggregate([simulate(s, p) for s in dev])
        raw = [simulate(s, raw_plan(row)) for s in dev]
        value = stats["mean_s_per_source"]
        rec = dict(id=row["id"], family=row["family"], params=row["params"],
            dev_s_per_source=value, raw_complete_rate=float(np.mean([r["complete"] for r in raw])),
            path_length_m=p["path_length_m"], stops=p["stops"], repair_count=p["repair_count"], selected=False)
        evaluated.append(rec)
        family = row["family"]
        if family not in optimized or value < optimized[family][0]:
            optimized[family] = (value, row)
        prepared_candidates[row["id"]] = p
        if (k+1) % 20 == 0 or k+1 == len(candidates):
            print(json.dumps(dict(stage="development", done=k+1, total=len(candidates),
                seconds=round(time.perf_counter()-started, 1))), flush=True)
    selected = list(bases)
    selected_ids = set()
    for family, (_, row) in optimized.items():
        row = dict(row)
        row["name"] = row["name"] + " · 参数优化"
        selected.append(row)
        selected_ids.add(row["id"])
    for rec in evaluated:
        rec["selected"] = rec["id"] in selected_ids
    plans = [prepared_candidates.get(r["id"]) or prepare_route(r) for r in selected]
    for p, row in zip(plans, selected):
        p["name"] = row["name"]
    scenes = [make_scene(test_start+i) for i in range(args.test_count)]
    summary, cases, raw_cases, raw_summary = [], {}, {}, []
    full_scan_summary = []
    for p in plans:
        rows = [dict(simulate(s, p), seed=test_start+i) for i, s in enumerate(scenes)]
        raw = [dict(simulate(s, raw_plan(p), scan_mode="all"), seed=test_start+i) for i, s in enumerate(scenes)]
        full = aggregate([simulate(s, p, scan_mode="all") for s in scenes])
        stats = aggregate(rows)
        raw_stats = aggregate(raw)
        stats.update(id=p["id"], name=p["name"], raw_complete_rate=raw_stats["complete_rate"],
                     raw_missed_total=raw_stats["missed_total"])
        summary.append(stats)
        raw_summary.append(dict(raw_stats, id=p["id"], name=p["name"]))
        full_scan_summary.append(dict(full, id=p["id"], name=p["name"]))
        cases[p["id"]], raw_cases[p["id"]] = rows, raw
        print(json.dumps(dict(stage="validation", id=p["id"], mean=round(stats["mean_s_per_source"], 3),
                             complete=stats["complete_rate"], raw=stats["raw_complete_rate"])), flush=True)

    # A single shared held-out group, with several explicitly difficult replays.
    replay_tags = {test_start: ["验证组首场"]}
    top = sorted(summary, key=lambda s: s["mean_s_per_source"])
    focus_ids = list(dict.fromkeys([p["id"] for p in plans if p["family"] in ("zigzag", "wave")]
                                   + [top[0]["id"], "single_ring", "two_layer"]))
    for route_id in focus_ids:
        if route_id not in cases:
            continue
        worst = max(cases[route_id], key=lambda r: r["s_per_source"])
        repair = max(cases[route_id], key=lambda r: r["repair_s"])
        name = next(p["name"] for p in plans if p["id"] == route_id)
        replay_tags.setdefault(worst["seed"], []).append(name+"最慢")
        if repair["repair_s"] > 0:
            replay_tags.setdefault(repair["seed"], []).append(name+"补盲最多")
    misses = sorted(raw_cases.get("single_ring", []), key=lambda r: len(r["missed_channels"]), reverse=True)
    for r in misses[:2]:
        replay_tags.setdefault(r["seed"], []).append("原始单圈漏检"+str(len(r["missed_channels"]))+"源")
    scenarios, replays = [], {}
    for seed, tags in replay_tags.items():
        scene = scenes[seed-test_start]
        scenarios.append(dict(seed=seed, tags=tags, sources=[asdict(s) for s in scene]))
        replays[str(seed)] = {p["id"]: simulate(scene, p, keep_replay=True) for p in plans}

    geometry = {}
    for p in plans:
        raw_geo = geometric_map(p["base_points"])
        repaired = certificate_details(p["points"])
        repaired["points"] = p["points"]
        geometry[p["id"]] = dict(raw=raw_geo, repaired=repaired,
                                 raw_certified_fraction=raw_geo["sampled_covered_fraction"])
    stress = []
    for p in plans:
        raw = [simulate(s, raw_plan(p)) for s in stress_scenes()]
        fixed = [simulate(s, p) for s in stress_scenes()]
        stress.append(dict(id=p["id"], raw_complete_rate=float(np.mean([r["complete"] for r in raw])),
            complete_rate=float(np.mean([r["complete"] for r in fixed])),
            raw_missed_total=sum(len(r["missed_channels"]) for r in raw)))
    best_id = min(evaluated, key=lambda r: r["dev_s_per_source"])["id"]
    best_values = np.array([r["s_per_source"] for r in cases[best_id]])
    paired = []
    for p in plans:
        diff = best_values - [r["s_per_source"] for r in cases[p["id"]]]
        se = float(np.std(diff, ddof=1)/math.sqrt(len(diff))) if len(diff) > 1 else 0.0
        paired.append(dict(reference=p["id"], best_id=best_id, difference_s=float(diff.mean()),
                           ci95_normal=[float(diff.mean()-1.96*se), float(diff.mean()+1.96*se)]))
    payload = dict(meta=dict(scope="local_synthetic_search_discovery_and_absence_certificate_excludes_localization_and_clear",
        dev_seed_start=dev_start, dev_count=args.dev_count, test_seed_start=test_start, test_count=args.test_count,
        test_total_sources=sum(len(s) for s in scenes), frozen_best_id=best_id,
        optimization_candidates=len(candidates), elapsed_wall_s=time.perf_counter()-started,
        numpy_version=np.__version__, physics=dict(domain_radius_m=1800, receive_m=[1000,1500],
            directional_probability=.55, beam_deg=180, speed_mps=5, measure_s=5, switch_s=1),
        notes=["同一独立验证组；开发集选参后冻结。", "搜索秒/源分母是评估器真实总源数，包含漏检场景；不是完整清除成绩。",
               "正常停止依赖几何无源证书或16个频道已发现；不读取实际源总数。",
               "全部路线用同一25点已证明构造补盲；45点方格只作额外保障。发现后停止为搜索而重复扫描该频道。",
               "方向热图每个非零位置都是可构造漏检反例；采样为零不代表连续保证。"],
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__), BASE/"routes.py")}),
        routes=plans, summary=summary, cases=cases, raw_summary=raw_summary,
        full_scan_summary=full_scan_summary, raw_cases=raw_cases,
        scenarios=scenarios, replays=replays, geometry=geometry,
        optimization=evaluated, stress_summary=stress, paired=paired)
    (out/"benchmark.json").write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    (out/"summary.json").write_text(json.dumps(dict(meta=payload["meta"], summary=summary, paired=paired, stress=stress), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(dict(done=True, output=str(out), seconds=time.perf_counter()-started)), flush=True)


if __name__ == "__main__":
    main()

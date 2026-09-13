"""Paired local full-mission experiments; truth is evaluator-only.

The frozen old joint policy is never tuned here. Failed or incomplete runs have
no valid full-mission score and cannot win by stopping prematurely.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import gzip
import hashlib
import json
from pathlib import Path
import time
import traceback

import numpy as np

from question4.full_mission.simulator import Simulator

BASE = Path(__file__).resolve().parent
PROJECT = BASE.parents[1]
OUT = BASE / "outputs"
COST_KEYS = ("movement", "measure", "switch", "clear_success", "clear_failure")
LABELS = {"old_joint": "旧联合：固定搜索站与已知源重排",
          "dynamic": "动态未知域：不计跨源定位价值",
          "dynamic_joint": "动态未知域＋跨源联合测点",
          "previous_dynamic": "上一版动态联合", "free_joint": "自由补点＋定向接收恢复"}


def write_json(path, data, compact=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=None if compact else 2,
                              separators=(",", ":") if compact else None,
                              allow_nan=False) + "\n", encoding="utf-8")


def old_parameters():
    path = BASE.parent / "full_mission" / "outputs" / "frozen_parameters.json"
    return json.loads(path.read_text(encoding="utf-8"))["parameters"]["joint"]


def resolve_configs(specification):
    """Resolve all dataclass defaults before the first task is dispatched."""
    configs = {}
    for name, spec in specification.items():
        policy = spec.get("policy", "old_joint" if name == "old_joint" else
                          "dynamic" if name == "previous_dynamic" else "free_joint")
        if policy == "old_joint":
            params = old_parameters()
            if spec.get("parameters") and spec["parameters"] != params:
                raise ValueError("old_joint must use its existing frozen parameters unchanged")
        elif policy == "dynamic":
            from .frozen_dynamic.planner import Parameters
            params = asdict(Parameters(**spec.get("parameters", {})))
        elif policy == "free_joint":
            from .planner import Parameters
            params = asdict(Parameters(**spec.get("parameters", {})))
        elif policy == "tuned_joint":
            from question4.tuned_joint.planner import Parameters
            params = asdict(Parameters(**spec.get("parameters", {})))
        else:
            raise ValueError(f"Unknown policy {policy}")
        configs[name] = dict(policy=policy, parameters=params, label=spec.get("label", LABELS.get(name, name)))
    return configs


def coverage_artifact(policy):
    """Persist actual per-channel observations, never turn plans into evidence."""
    coverage = getattr(policy, "unknown_map", policy.coverage)
    points = getattr(coverage, "points", [])
    if callable(points):
        points = []  # Dynamic map has different actual sites per channel.
    artifact = {"search_points": [np.asarray(q).tolist() for q in points]}
    # Per-channel negative sites provide an independent observation record even
    # if the dynamic map's internal representation changes.
    artifact["negative_points_by_channel"] = {
        str(c): [o["position"] for o in t.observations if o.get("result") == "no_signal"]
        for c, t in policy.targets.items()
    }
    if hasattr(coverage, "snapshot"):
        try:
            artifact["coverage"] = coverage.snapshot()
        except TypeError:
            pass
    elif hasattr(coverage, "_proofs"):
        # The old map stores monotone proof supports shared by unknown channels.
        artifact["coverage"] = {"certificate_supports": {
            str(c): [p["points"].tolist() for p in coverage._proofs
                     if p["support"] <= {(float(q[0]) or 0., float(q[1]) or 0.)
                                         for q in artifact["negative_points_by_channel"][str(c)]}]
            for c in policy.targets}}
    return artifact


def run_one(job):
    seed, name, spec, directory, record = job
    sim = Simulator(seed)
    truth = sim.truth()
    policy = None
    started = time.perf_counter()
    error = None
    try:
        if spec["policy"] == "old_joint":
            from question4.full_mission.planner import Parameters, Planner
        elif spec["policy"] == "dynamic":
            from .frozen_dynamic.planner import Parameters, Planner
        elif spec["policy"] == "tuned_joint":
            from question4.tuned_joint.planner import Parameters, Planner
        else:
            from .planner import Parameters, Planner
        policy = Planner(sim.command, Parameters(**spec["parameters"]), record=record)
        result = policy.run()
        if not result.get("completed") or not all(s.cleared for s in sim.sources):
            raise AssertionError("Public policy stopped before all real sources were cleared")
        if len(policy.commands) != len(sim.history):
            raise AssertionError("Public policy command count disagrees with simulator")
        if any(abs(result["costs"][k] - sim.costs[k]) > 1e-6 for k in COST_KEYS):
            raise AssertionError("Independent fee ledgers disagree")
        if abs(sum(sim.costs.values()) - sim.time_s) > 1e-6 or abs(result["time_s"] - sim.time_s) > 1e-6:
            raise AssertionError("Total time differs from actual fee ledger")
    except Exception:
        error = traceback.format_exc()
        result = dict(time_s=sim.time_s, costs=dict(sim.costs), completed=False,
                      counters=getattr(policy, "counters", {}),
                      cleared=sum(s.cleared for s in sim.sources),
                      source_times=getattr(policy, "target_times", {}),
                      command_count=len(getattr(policy, "commands", [])),
                      coverage_certified=bool(getattr(getattr(policy, "coverage", None), "certified", False)),
                      parameters=spec["parameters"])
    n = len(truth)
    completed = bool(result.get("completed") and error is None)
    result.update(seed=seed, strategy=name, label=spec["label"], policy=spec["policy"],
                  source_count=n, completed=completed, per_source_s=sim.time_s / n if completed else None,
                  elapsed_per_source_s=sim.time_s / n, error=error, wall_s=time.perf_counter() - started)
    result["costs_per_source"] = {k: v / n for k, v in sim.costs.items()}
    commands = getattr(policy, "commands", [])
    by_phase = {}
    for row in commands:
        bucket = by_phase.setdefault(row["phase"], dict(time_s=0., move_s=0., commands=0))
        bucket["time_s"] += row["move_s"] + row["action_s"]
        bucket["move_s"] += row["move_s"]
        bucket["commands"] += 1
    result["phases"] = by_phase
    source_times = result.get("source_times", {})
    result["max_source_service_s"] = max((r.get("service_time_s", 0.) for r in source_times.values()), default=0.)
    result["search_after_last_clear_s"] = sim.time_s - max((r.get("cleared_s") or 0. for r in source_times.values()), default=0.)
    frames = getattr(policy, "frames", []) if record else None
    history = dict(summary=result, truth=truth, commands=commands, frames=frames)
    if policy is not None:
        history.update(coverage_artifact(policy))
    if directory:
        path = Path(directory) / name / f"{seed}.json.gz"
        path.parent.mkdir(parents=True, exist_ok=True)
        result["history_file"] = str(path.resolve())
        with gzip.open(path, "wt", encoding="utf-8") as stream:
            json.dump(history, stream, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return dict(summary=result, frames=frames, truth=truth) if record else result


def aggregate(rows, name):
    all_rows = [r for r in rows if r["strategy"] == name]
    successful = [r for r in all_rows if r["completed"] and r["per_source_s"] is not None]
    values = np.asarray([r["per_source_s"] for r in successful])
    complete = len(successful) == len(all_rows)
    def avg(items): return float(np.mean(items)) if items else None
    phases = sorted({k for r in all_rows for k in r["phases"]})
    counters = sorted({k for r in all_rows for k in r["counters"]})
    return dict(id=name, strategy=name, label=all_rows[0]["label"], count=len(all_rows),
                source_count=sum(r["source_count"] for r in all_rows), cleared=sum(r["cleared"] for r in all_rows),
                completed_count=len(successful), failed_count=len(all_rows) - len(successful),
                completion_rate=len(successful) / len(all_rows),
                mean_per_source_s=float(values.mean()) if complete else None,
                p90_per_source_s=float(np.quantile(values, .9)) if complete else None,
                worst_per_source_s=float(values.max()) if complete else None,
                median_per_source_s=float(np.median(values)) if complete else None,
                completed_only_mean_per_source_s=float(values.mean()) if len(values) else None,
                pooled_per_source_s=sum(r["time_s"] for r in all_rows) / sum(r["source_count"] for r in all_rows) if complete else None,
                mean_total_s=avg([r["time_s"] for r in all_rows]) if complete else None,
                costs_per_source={k: avg([r["costs"][k] / r["source_count"] for r in all_rows]) for k in COST_KEYS},
                mean_counters={k: avg([r["counters"].get(k, 0.) for r in all_rows]) for k in counters},
                phase_costs_per_source={k: avg([r["phases"].get(k, {}).get("time_s", 0.) / r["source_count"] for r in all_rows]) for k in phases},
                mean_search_after_last_clear_s=avg([r["search_after_last_clear_s"] for r in all_rows]),
                mean_planning_wall_s=avg([r.get("planning_wall_s", r.get("wall_s", 0.)) for r in all_rows]),
                worst_planning_wall_s=max(r.get("planning_wall_s", r.get("wall_s", 0.)) for r in all_rows),
                worst_seed=max(successful, key=lambda r: r["per_source_s"])["seed"] if successful else None,
                failed_seeds=[r["seed"] for r in all_rows if not r["completed"]],
                score_rule="Full-mission statistics are null if any run failed; costs on failed runs are elapsed costs only")


def paired(rows, first, second):
    a = {r["seed"]: r for r in rows if r["strategy"] == first}
    b = {r["seed"]: r for r in rows if r["strategy"] == second}
    shared = sorted(a.keys() & b.keys())
    complete = [s for s in shared if a[s]["completed"] and b[s]["completed"]]
    difference = np.asarray([a[s]["per_source_s"] - b[s]["per_source_s"] for s in complete])
    result = dict(first=first, second=second, meaning="first minus second, seconds/source",
                  paired_count=len(shared), complete_pairs=len(complete), failed_pairs=len(shared) - len(complete),
                  complete_group=len(complete) == len(shared), mean_difference_s=None, bootstrap95_s=None,
                  first_faster_fraction=None)
    if len(difference):
        rng = np.random.default_rng(419)
        boot = difference[rng.integers(0, len(difference), (5000, len(difference)))].mean(axis=1)
        result.update(mean_difference_s=float(difference.mean()), bootstrap95_s=np.quantile(boot, [.025, .975]).tolist(),
                      first_faster_fraction=float(np.mean(difference < 0)))
    return result


def code_hashes():
    paths = [*BASE.glob("*.py"), *(BASE / "frozen_dynamic").glob("*.py"),
             *(BASE.parent / "tuned_joint").glob("planner.py"), BASE.parent / "full_mission" / "planner.py",
             BASE.parent / "full_mission" / "service.py", BASE.parent / "full_mission" / "simulator.py",
             BASE.parent / "zigzag_study" / "study.py", BASE.parent / "zigzag_study" / "routes.py",
             BASE.parent / "benchmark.py", PROJECT / "question1" / "geometry.py",
             PROJECT / "question1" / "enclosing_circle.py"]
    return {str(p.relative_to(PROJECT)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in paths if p.name not in ("evaluate.py", "audit.py", "replay.py", "diagnostics.py", "behavior_analysis.py") and not p.name.startswith("test_")}


def batch(configs, seeds, directory, workers=6, record=False):
    started = time.perf_counter()
    jobs = [(s, name, spec, str(directory), record) for name, spec in configs.items() for s in seeds]
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(run_one, job): job[:2] for job in jobs}
        for future in as_completed(futures):
            result = future.result()
            rows.append(result)
            r = result["summary"] if record else result
            if r.get("error"):
                print(json.dumps(dict(failed=futures[future], error=r["error"]), ensure_ascii=False), flush=True)
            if len(rows) % 5 == 0 or len(rows) == len(jobs):
                print(json.dumps(dict(done=len(rows), total=len(jobs), wall_s=round(time.perf_counter() - started, 1))), flush=True)
    rows.sort(key=lambda r: ((r.get("summary") or r)["strategy"], (r.get("summary") or r)["seed"]))
    return rows


def experiment(stage, specs, seeds, output, workers):
    configs = resolve_configs(specs)
    before = code_hashes()
    rows = batch(configs, seeds, output / f"{stage}_histories", workers)
    after = code_hashes()
    if before != after:
        write_json(output / f"{stage}_code_changed.json", dict(before=before, after=after, rows=rows))
        raise RuntimeError("Policy code changed during batch; results saved but cannot be treated as frozen")
    names = list(configs)
    summary = [aggregate(rows, name) for name in names]
    comparisons = [paired(rows, a, b) for i, a in enumerate(names) for b in names[:i]]
    data = dict(stage=stage, scope="local synthetic complete mission", seeds=seeds, configs=configs,
                summary=summary, rows=rows, paired=comparisons, code_sha256=after)
    write_json(output / f"{stage}.json", data)
    print(json.dumps(dict(summary=summary, paired=comparisons), ensure_ascii=False), flush=True)
    return data


def make_replays(validation, output, workers, seeds=None):
    rows = validation["rows"]
    selected = {}
    def tag(seed, text): selected.setdefault(int(seed), []).append(text)
    if seeds:
        for seed in seeds: tag(seed, "指定复核样本")
    else:
        current = {r["seed"]: r for r in rows if r["strategy"] == "free_joint" and r["completed"]}
        old = {r["seed"]: r for r in rows if r["strategy"] == "old_joint" and r["completed"]}
        shared = current.keys() & old.keys()
        if shared:
            tag(max(current, key=lambda s: current[s]["per_source_s"]), "动态联合每源耗时最长")
            for sign, text in ((1, "动态联合相对旧联合退步最多"), (-1, "动态联合相对旧联合进步最多")):
                tag(max(shared, key=lambda s: sign * (current[s]["per_source_s"] - old[s]["per_source_s"])), text)
            eligible = set(current) - set(selected) or set(current)
            tag(max(eligible, key=lambda s: current[s]["counters"].get("actual_cross_source_measurements", 0)), "动态联合跨源补测活跃样本")
        else:
            for name, spec in validation["configs"].items():
                successful = [r for r in rows if r["strategy"] == name and r["completed"]]
                if successful:
                    tag(max(successful, key=lambda r: r["per_source_s"])["seed"], f"{spec['label']}：最慢")
    result = batch(validation["configs"], sorted(selected), output / "replay_histories", workers, record=True)
    original = {(r["strategy"], r["seed"]): r for r in rows}
    replays = {str(s): {} for s in sorted(selected)}
    scenes = {}
    for r in result:
        summary = r["summary"]
        prior = original[(summary["strategy"], summary["seed"])]
        if (abs(summary["time_s"] - prior["time_s"]) > 1e-7
                or summary["command_count"] != prior["command_count"] or summary["completed"] != prior["completed"]):
            raise AssertionError("Detailed recording changed executed actions relative to benchmark")
        key = str(summary["seed"])
        scenes[key] = dict(seed=summary["seed"], tags=selected[summary["seed"]], truth=r.pop("truth"))
        replays[key][summary["strategy"]] = r
    meta = dict(strategies=[dict(id=k, label=v["label"]) for k, v in validation["configs"].items()],
                test_count=len(validation["seeds"]),
                description=f"本地完整任务，共同场景 {len(validation['seeds'])} 场；种子 {validation['seeds'][0]}–{validation['seeds'][-1]}。主指标为各场完整总时间/实际源数的等权平均。",
                limitations=["全部本地合成；总时间包括移动、测量、切频、成功与失败清除",
                             "失联不等于附近无源，离散未知域只指导测点；停止依赖连续覆盖证书",
                             "预期收益为有限候选启发式；差案例不是总体成绩，未连接官方服务"])
    data = dict(meta=meta, summary=validation["summary"], paired=validation["paired"],
                scenarios=list(scenes.values()), replays=replays)
    write_json(output / "data.json", data, compact=True)
    from .replay import render
    (output / "report.html").write_text(render(data), encoding="utf-8")
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("smoke", "develop", "validate", "replays"), required=True)
    parser.add_argument("--variants", type=Path)
    parser.add_argument("--seed-start", type=int)
    parser.add_argument("--seeds", help="Explicit comma-separated seeds for labelled development regressions")
    parser.add_argument("--count", type=int)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--freeze-from", help="After development, freeze this complete configuration as dynamic_joint")
    parser.add_argument("--replay-seeds", help="Comma-separated seeds already present in validate.json")
    args = parser.parse_args()
    if args.stage == "replays":
        validation = json.loads((args.output / "validate.json").read_text(encoding="utf-8"))
        make_replays(validation, args.output, args.workers,
                     [int(s) for s in args.replay_seeds.split(",")] if args.replay_seeds else None)
        return
    if args.variants:
        specs = json.loads(args.variants.read_text(encoding="utf-8"))
        specs = specs.get("configs", specs)
    elif args.stage == "validate":
        specs = json.loads((args.output / "frozen_parameters.json").read_text(encoding="utf-8"))["configs"]
    else:
        specs = {"old_joint": {"policy": "old_joint"}, "free_joint": {"policy": "free_joint"}}
    start = args.seed_start if args.seed_start is not None else 20286000 if args.stage == "validate" else 20285000
    count = args.count or {"smoke": 2, "develop": 16, "validate": 100}[args.stage]
    if count < 1:
        parser.error("count must be positive")
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else list(range(start, start + count))
    data = experiment(args.stage, specs, seeds, args.output, args.workers)
    if args.freeze_from:
        selected = next(r for r in data["summary"] if r["id"] == args.freeze_from)
        if (args.stage != "develop" or selected["failed_count"]
                or data["configs"][args.freeze_from]["policy"] != "free_joint"):
            raise ValueError("Freeze only a fully completed development configuration")
        configs = {"old_joint": resolve_configs({"old_joint": {"policy": "old_joint"}})["old_joint"],
                   "free_joint": {**data["configs"][args.freeze_from], "label": LABELS["free_joint"]}}
        write_json(args.output / "frozen_parameters.json", dict(selected_on_development=args.freeze_from,
                   development_seeds=data["seeds"], configs=configs))


if __name__ == "__main__":
    main()

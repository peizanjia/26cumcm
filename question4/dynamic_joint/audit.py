"""Reconstruct physical outcomes and charges from saved evaluator histories.

No simulator or policy execution is used. Coverage is independently checked
against actual per-channel negative observations, not probability-grid mass.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path

from question4.full_mission.audit_results import close, point_key, require, truth_signature
from question4.zigzag_study.study import certificate_details

BASE = Path(__file__).resolve().parent
PROJECT = BASE.parents[1]
COST_KEYS = ("movement", "measure", "switch", "clear_success", "clear_failure")


def audit_history(history, outer):
    summary, commands, truth = history["summary"], history["commands"], history["truth"]
    require(summary["completed"] and not summary.get("error"), "Saved run incomplete or failed")
    require(len(truth) == summary["source_count"], "Source count mismatch")
    sources = {int(t["channel"]): t for t in truth}
    require(len(sources) == len(truth), "Source channels duplicated")
    require(all(not t["cleared"] for t in truth), "Expected initial uncleared truth")
    cleared, known = set(), set()
    negatives, optical_sites = defaultdict(set), defaultdict(set)
    repeats, costs = {}, {k: 0. for k in COST_KEYS}
    phases = defaultdict(lambda: dict(time_s=0., move_s=0., commands=0))
    counts = dict(measures=0, clears=0, clear_failures=0, no_signal=0, known_no_signal=0, optical_fallback_clears=0)
    source_times = {c: dict(found_s=None, cleared_s=None, service_time_s=0., service_move_m=0., measures=0,
                            clears=0, failed_clears=0) for c in range(1, 21)}
    position, channel, elapsed = (0., 0.), 1, 0.
    duplicate_reads = 0
    for i, row in enumerate(commands):
        context = f"action {i}"
        require(row["index"] == i, f"{context}: nonsequential index")
        require(point_key(row["from"]) == point_key(position), f"{context}: wrong movement origin")
        q, c, kind = point_key(row["position"]), int(row["channel"]), row["kind"]
        require(len(q) == 2 and all(math.isfinite(v) for v in q), f"{context}: invalid point")
        require(1 <= c <= 20 and kind in ("measure", "clear"), f"{context}: invalid command")
        response = row["response"]
        require(response.get("accepted") is True, f"{context}: unaccepted command")
        decision = row.get("decision") or {}
        selected = decision.get("selected")
        if isinstance(selected, dict) and selected.get("position") is not None:
            require(selected.get("kind") == kind and int(selected.get("channel")) == c
                    and math.dist(selected["position"], q) <= 1e-7,
                    f"{context}: executed action differs from selected candidate")
            selected_score = selected.get("selection_s", selected.get("score_s"))
            scores = [r.get("selection_s", r.get("score_s")) for r in decision.get("candidates", [])]
            scores = [float(v) for v in scores if v is not None]
            if selected_score is not None and scores:
                require(all(math.isfinite(v) for v in scores), f"{context}: nonfinite candidate score")
                close(selected_score, min(scores), f"{context}: selected candidate is not minimum recorded cost")
        source = sources.get(c) if c not in cleared else None
        distance = math.hypot(q[0] - source["x"], q[1] - source["y"]) if source else math.inf
        movement = math.dist(position, q) / 5.
        switching, before = 0., elapsed
        state = source_times[c]
        if kind == "measure":
            fee = 5.
            switching = float(c != channel)
            channel = c
            costs["measure"] += fee
            costs["switch"] += switching
            counts["measures"] += 1
            state["measures"] += 1
            illuminated = bool(source and (not source["directional"] or
                (q[0] - source["x"]) * math.cos(source["orientation"]) +
                (q[1] - source["y"]) * math.sin(source["orientation"]) >= -1e-9))
            expected = ("no_signal" if source is None or not illuminated or distance > source["radius"]
                        else "near" if distance <= 5. else "direction")
            require(response.get("measure_result") == expected, f"{context}: incorrect range/orientation outcome")
            if expected == "direction":
                exact = math.degrees(math.atan2(source["y"] - q[1], source["x"] - q[0]))
                error = abs((response["svd_deg"] - exact + 180.) % 360. - 180.)
                require(error <= 1.00500001, f"{context}: bearing outside error interval")
            key = c, q
            if c not in cleared:
                value = expected, response.get("svd_deg")
                if key in repeats:
                    duplicate_reads += 1
                    require(repeats[key] == value, f"{context}: repeated-site signal changed")
                repeats[key] = value
            if expected == "no_signal":
                negatives[c].add(q)
                counts["no_signal"] += 1
                counts["known_no_signal"] += int(c in known and c not in cleared)
            else:
                known.add(c)
        else:
            success = source is not None and distance <= 20.
            require(response.get("clear_result") == ("success" if success else "no_target_in_range"),
                    f"{context}: optical clear outcome violates true distance")
            fee = 5. if success else 3.
            costs["clear_success" if success else "clear_failure"] += fee
            counts["clears"] += 1
            counts["clear_failures"] += int(not success)
            state["clears"] += 1
            state["failed_clears"] += int(not success)
            if success: cleared.add(c)
            if row.get("reason") == "optical_finite_cover":
                require(q not in optical_sites[c], f"{context}: repeated optical fallback stop")
                optical_sites[c].add(q)
                require(len(optical_sites[c]) <= 122, f"{context}: finite optical cover exhausted")
                counts["optical_fallback_clears"] += 1
        costs["movement"] += movement
        elapsed += movement + fee + switching
        close(row["move_s"], movement, f"{context}: movement cost")
        close(row["action_s"], fee + switching, f"{context}: action/switch cost")
        close(row["time_s"], elapsed, f"{context}: clock")
        close(response["virtual_time_s"], elapsed, f"{context}: public clock")
        for key in COST_KEYS: close(row["costs"][key], costs[key], f"{context}: {key}")
        if response.get("measure_result") in ("near", "direction") and state["found_s"] is None:
            state["found_s"] = elapsed
        if response.get("clear_result") == "success": state["cleared_s"] = elapsed
        if row["phase"] in ("service", "side_known"):
            state["service_time_s"] += elapsed - before
            state["service_move_m"] += movement * 5.
        phase = phases[row["phase"]]
        phase["time_s"] += movement + fee + switching
        phase["move_s"] += movement
        phase["commands"] += 1
        position = q
    require(cleared == set(sources), "Stopped with real sources uncleared")
    require(len(commands) == summary["command_count"], "Command count mismatch")
    require(summary["cleared"] == len(cleared), "Clear count mismatch")
    require(len(cleared) == 16 or summary["coverage_certified"], "No valid public termination flag")
    if summary["coverage_certified"]:
        absent = set(range(1, 21)) - set(sources)
        for c in absent:
            require(len(negatives[c]) >= 3, f"Absent channel {c}: insufficient measurements")
        # All negative observations are actual evidence. A remembered support
        # can pass even if insertion changed the full Delaunay triangulation.
        supports = history.get("coverage", {}).get("certificate_supports", {})
        for c in absent:
            points = [list(q) for q in negatives[c]]
            proof = certificate_details(points)
            if not proof["certified"]:
                options = supports.get(str(c), [])
                channel_snapshot = history.get("coverage", {}).get("channels", {}).get(str(c), {})
                if channel_snapshot.get("proof_points"):
                    options = [*options, channel_snapshot["proof_points"]]
                proof_ok = False
                for support in options:
                    support = support.get("points", support) if isinstance(support, dict) else support
                    require(all(point_key(q) in negatives[c] for q in support), f"Absent channel {c}: unobserved certificate support")
                    if certificate_details(support)["certified"]:
                        proof_ok = True
                        break
                # Old fixed policy's common observed search set is also valid.
                old_points = history.get("search_points", [])
                if old_points and all(point_key(q) in negatives[c] for q in old_points):
                    proof_ok = proof_ok or certificate_details(old_points)["certified"]
                require(proof_ok, f"Absent channel {c}: cannot independently certify observed coverage")
    for c, points in history.get("negative_points_by_channel", {}).items():
        require({point_key(q) for q in points} == negatives[int(c)], f"Channel {c}: negative-map history differs from actual measurements")
    frames = history.get("frames") or []
    if frames:
        require(len(frames) == len(commands), "Detailed replay frame count differs from actual commands")
    for frame in frames:
        for target in frame.get("targets", []):
            polygon = target.get("polygon")
            c = int(target["channel"])
            if not polygon or c not in sources or target["status"] not in ("active", "cleared"):
                continue
            source = sources[c]
            signs = []
            for p, q in zip(polygon, polygon[1:] + polygon[:1]):
                length = max(1e-12, math.dist(p, q))
                signs.append(((q[0] - p[0]) * (source["y"] - p[1])
                              - (q[1] - p[1]) * (source["x"] - p[0])) / length)
            require(min(signs) >= -2e-4 or max(signs) <= 2e-4,
                    f"Replay action {frame['index']} channel {c}: conservative polygon lost real source")
    close(summary["time_s"], elapsed, "Final time")
    close(summary["per_source_s"], elapsed / len(sources), "Full time/source")
    close(sum(costs.values()), elapsed, "Final fee sum")
    for key in COST_KEYS:
        close(summary["costs"][key], costs[key], f"Final cost {key}")
        close(summary["costs_per_source"][key], costs[key] / len(sources), f"Per-source cost {key}")
    for key, value in counts.items():
        require(summary["counters"].get(key) == value, f"Counter {key} mismatch")
    for c, state in source_times.items():
        saved = summary["source_times"].get(str(c), summary["source_times"].get(c))
        for key, value in state.items():
            if value is None: require(saved[key] is None, f"Channel {c}: spurious {key}")
            else: close(saved[key], value, f"Channel {c}: {key}")
    for phase, values in phases.items():
        for key, value in values.items(): close(summary["phases"][phase][key], value, f"Phase {phase}: {key}")
    for key in ("time_s", "per_source_s", "command_count", "source_count", "cleared"):
        close(outer[key], summary[key], f"Outer summary differs: {key}")
    return dict(strategy=summary["strategy"], seed=summary["seed"], sources=len(sources),
                clears=len(cleared), commands=len(commands), repeated_measurements=duplicate_reads)


def audit(validation_path, output_path, policy_archive=None):
    path = Path(validation_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    results, issues, scenes, seen = [], [], {}, set()
    for row in data["rows"]:
        key = row["strategy"], row["seed"]
        try:
            require(key not in seen, "Duplicate paired row")
            seen.add(key)
            with gzip.open(row["history_file"], "rt", encoding="utf-8") as stream:
                history = json.load(stream)
            signature = truth_signature(history["truth"])
            require(row["seed"] not in scenes or scenes[row["seed"]] == signature, "Paired policies saw different scenes")
            scenes[row["seed"]] = signature
            results.append(audit_history(history, row))
        except Exception as error:
            issues.append(dict(strategy=row["strategy"], seed=row["seed"], error=f"{type(error).__name__}: {error}"))
    expected = {(name, seed) for name in data["configs"] for seed in data["seeds"]}
    if seen != expected: issues.append(dict(error="Incomplete paired matrix", missing=sorted(expected - seen), extra=sorted(seen - expected)))
    for name, digest in data.get("code_sha256", {}).items():
        source = PROJECT / name
        if policy_archive is not None and name.startswith("question4/dynamic_joint/"):
            source = Path(policy_archive) / Path(name).name
        if not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest() != digest:
            issues.append(dict(error="Policy source changed after experiment", path=name))
    result = dict(passed=not issues, validation=str(path.resolve()), audited_runs=len(results),
                  policy_archive=str(Path(policy_archive).resolve()) if policy_archive else None,
                  unique_scenarios=len(scenes), commands=sum(r["commands"] for r in results),
                  physical_sources=sum(len(json.loads(v)) for v in scenes.values()),
                  clears_across_runs=sum(r["clears"] for r in results),
                  repeated_measurements=sum(r["repeated_measurements"] for r in results), issues=issues,
                  checks=["independent 180-degree reception and <=20m optical outcomes", "all five cost categories and receiver channel state",
                          "fixed repeated-site bearing feedback", "per-channel negative coverage evidence and continuous geometric certificate",
                          "complete actual clearing", "paired scene equality", "persisted command/summary/source/phase accounting", "frozen code hashes"])
    Path(output_path).write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation", type=Path, default=BASE / "outputs" / "validate.json")
    parser.add_argument("--output", type=Path, default=BASE / "outputs" / "audit.json")
    parser.add_argument("--policy-archive", type=Path,
                        help="Verify archived dynamic policy hashes while reauditing a preserved earlier batch")
    args = parser.parse_args()
    result = audit(args.validation, args.output, args.policy_archive)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()

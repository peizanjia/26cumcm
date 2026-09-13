"""Matérn-5/2 Gaussian-process / expected-improvement parameter search.

This reuses the read-only SciPy GP implementation used in question 3. Every
candidate runs on the same training scenes. A result with any unfinished scene
receives a failure penalty and cannot be selected as the best complete policy.
The default budget is one baseline, eight Sobol proposals and 24 GP/EI proposals.
This is finite-budget parameter selection, never a proof of global optimality.

Example space JSON::

    {"forecast_weight": {"type": "float", "low": 0.7, "high": 1.3},
     "max_radio_steps": {"type": "int", "low": 6, "high": 12}}

The evaluator module must expose resolve_configs, run_one and code_hashes using
question4.free_joint.evaluate's interface. In that evaluator only sim.command
is passed to the planner; simulator truth remains an evaluation/audit input.
Code must stay unchanged during a search, including resumes. Extend --trials or
change --workers when resuming, but do not change the seeds, bounds or baseline.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import importlib
import inspect
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
import traceback

import numpy as np
import scipy
from scipy.stats import qmc

from question3.global_policy.tour_optimization.gaussian_process import (
    GaussianProcess, expected_improvement,
)


FORMAT_VERSION = 1
COST_KEYS = ("movement", "measure", "switch", "clear_success", "clear_failure")


def write_json(path, value):
    """Replace a complete JSON file atomically; interrupted writes stay separate."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


class SearchSpace:
    """A normalized box with canonical rounding of integer coordinates."""

    def __init__(self, specification):
        if not isinstance(specification, dict) or not specification:
            raise ValueError("Search space must be a nonempty object keyed by parameter name")
        self.dimensions = []
        for name, item in specification.items():
            if isinstance(item, (list, tuple)) and len(item) == 2:
                item = dict(type="float", low=item[0], high=item[1])
            kind = item.get("type", "float")
            if kind in ("continuous", "real"):
                kind = "float"
            if kind == "integer":
                kind = "int"
            if kind not in ("float", "int"):
                raise ValueError(f"{name}: only float and int dimensions are supported")
            lo, hi = float(item["low"]), float(item["high"])
            scale = item.get("scale", "linear")
            if not (math.isfinite(lo) and math.isfinite(hi) and lo < hi):
                raise ValueError(f"{name}: finite low < high bounds are required")
            if scale not in ("linear", "log") or (scale == "log" and lo <= 0):
                raise ValueError(f"{name}: invalid scale or nonpositive logarithmic bound")
            if kind == "int" and (not lo.is_integer() or not hi.is_integer()):
                raise ValueError(f"{name}: integer dimensions need integer bounds")
            self.dimensions.append(dict(name=name, type=kind, low=lo, high=hi, scale=scale))
        self.specification = {d["name"]: {k: v for k, v in d.items() if k != "name"}
                              for d in self.dimensions}

    def __len__(self):
        return len(self.dimensions)

    def encode(self, parameters):
        result = []
        for d in self.dimensions:
            if d["name"] not in parameters:
                raise ValueError(f"Missing parameter in resolved baseline: {d['name']}")
            value = float(parameters[d["name"]])
            lo, hi = d["low"], d["high"]
            if not math.isfinite(value) or not lo <= value <= hi:
                raise ValueError(f"{d['name']}: baseline value {value} outside [{lo}, {hi}]")
            if d["type"] == "int" and not value.is_integer():
                raise ValueError(f"{d['name']}: integer coordinate has noninteger value")
            if d["scale"] == "log":
                value, lo, hi = np.log([value, lo, hi])
            result.append((value - lo) / (hi - lo))
        return np.asarray(result)

    def decode(self, vector):
        vector = np.asarray(vector, dtype=float)
        if vector.shape != (len(self),) or not np.isfinite(vector).all():
            raise ValueError("A finite normalized coordinate per dimension is required")
        result = {}
        for coordinate, d in zip(np.clip(vector, 0., 1.), self.dimensions):
            lo, hi = d["low"], d["high"]
            if d["scale"] == "log":
                value = float(np.exp(np.log(lo) + coordinate * (np.log(hi) - np.log(lo))))
            else:
                value = float(lo + coordinate * (hi - lo))
            value = min(hi, max(lo, value))
            result[d["name"]] = int(round(value)) if d["type"] == "int" else value
        return result

    def canonical(self, vector):
        return self.encode(self.decode(vector))


def resolve_baseline(evaluator, specification):
    """Accept a policy spec, one-entry configs mapping, or saved configs wrapper."""
    if "configs" in specification:
        specification = specification["configs"]
    if "policy" in specification:
        specification = {"baseline": specification}
    if len(specification) != 1:
        raise ValueError("--base-config must contain exactly one strategy")
    resolved = evaluator.resolve_configs(specification)
    if len(resolved) != 1:
        raise ValueError("Evaluator did not resolve exactly one baseline strategy")
    return next(iter(resolved.values()))


def source_hashes(evaluator):
    result = dict(evaluator.code_hashes())
    # The evaluated algorithm, evaluator and proposal implementation are guarded.
    # The evaluator owns discovery of any additional planner runtime dependencies.
    for module in (sys.modules[__name__], evaluator, inspect.getmodule(GaussianProcess)):
        filename = getattr(module, "__file__", None)
        if filename:
            path = Path(filename).resolve()
            result[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def check_sources(evaluator, expected, output):
    actual = source_hashes(evaluator)
    if actual != expected:
        write_json(Path(output) / "code_changed.json", dict(
            expected=expected, actual=actual,
            reason="Runtime/evaluator/search code changed; results are preserved but this search cannot resume"))
        raise RuntimeError("Source hash guard failed; see code_changed.json. Use a new experiment directory.")


def summarize(rows, seeds, failure_penalty=1e6, baseline_rows=None):
    """Keep all scenes and all charged costs; incomplete trials are ineligible."""
    seeds = list(seeds)
    by_seed = {int(row["seed"]): row for row in rows}
    if len(by_seed) != len(rows) or set(by_seed) != set(seeds):
        raise ValueError("A trial must contain each common training seed exactly once")
    values, elapsed, failed = [], [], []
    for seed in seeds:
        row = by_seed[seed]
        value = row.get("per_source_s")
        n = int(row.get("source_count", 0))
        valid = (bool(row.get("completed")) and value is not None and n > 0
                 and math.isfinite(float(value)) and float(value) >= 0
                 and ("cleared" not in row or int(row["cleared"]) == n))
        if valid and "costs" in row and "time_s" in row:
            valid = (abs(sum(row["costs"].values()) - row["time_s"]) < 1e-5
                     and abs(float(value) * n - row["time_s"]) < 1e-5)
        amount = float(row.get("elapsed_per_source_s", value or 0.))
        amount = amount if math.isfinite(amount) and amount >= 0 else 0.
        elapsed.append(amount)
        if valid:
            values.append(float(value))
        else:
            failed.append(seed)
    complete = not failed
    objective = (float(np.mean(values)) if complete else
                 failure_penalty * (1 + len(failed)/len(seeds)) + float(np.mean(elapsed)))
    delta = objective
    variance = 0.
    if baseline_rows is not None:
        base = summarize(baseline_rows, seeds, failure_penalty)
        delta -= base["objective"]
        if complete and base["complete"] and len(seeds) > 1:
            lookup = {row["seed"]: row for row in baseline_rows}
            paired = [by_seed[s]["per_source_s"] - lookup[s]["per_source_s"] for s in seeds]
            variance = float(np.var(paired, ddof=1)/len(seeds))
    elif complete and len(seeds) > 1:
        variance = float(np.var(values, ddof=1)/len(seeds))
    cost_means = {}
    for key in COST_KEYS:
        valid_costs = [float(r.get("costs", {}).get(key, 0.))/r["source_count"]
                       for r in rows if r.get("source_count", 0) > 0]
        cost_means[key] = float(np.mean(valid_costs)) if valid_costs else None
    return dict(objective=objective, delta_from_baseline_s=delta,
                paired_mean_variance=variance, complete=complete,
                count=len(seeds), failed_count=len(failed), failed_seeds=failed,
                source_count=sum(r.get("source_count", 0) for r in rows),
                cleared=sum(r.get("cleared", 0) for r in rows),
                mean_per_source_s=float(np.mean(values)) if complete else None,
                p90_per_source_s=float(np.quantile(values, .9)) if complete else None,
                mean_elapsed_per_source_s=float(np.mean(elapsed)),
                mean_costs_per_source=cost_means,
                mean_planning_wall_s=float(np.mean([r.get("planning_wall_s", r.get("wall_s", 0.)) for r in rows])),
                note="All scenes retained; elapsed costs on failed scenes do not describe a complete mission")


def _unique_pool(space, vectors, seen):
    points, used = [], set(seen)
    for vector in vectors:
        point = space.canonical(vector)
        key = tuple(np.round(point, 12))
        if key not in used:
            points.append(point)
            used.add(key)
    return np.asarray(points)


def propose(space, trials, initial, seed, pool_power=11):
    """Deterministic Sobol warm-up followed by a fitted GP/EI proposal."""
    seen = {tuple(np.round(space.encode(t["searched_parameters"]), 12)) for t in trials}
    initial_done = sum(t["proposal"]["kind"] == "sobol" for t in trials)
    if initial_done < initial:
        vectors = qmc.Sobol(d=len(space), scramble=True, seed=seed).random_base2(
            max(4, int(math.ceil(math.log2(initial * 4 + 1)))))
        pool = _unique_pool(space, vectors, seen)
        if len(pool) == 0:
            raise ValueError("Search space exhausted before the requested unique Sobol warm-up")
        return pool[0], dict(kind="sobol", index=initial_done, random_seed=seed)
    finished = [t for t in trials if t["status"] == "complete"]
    x = np.asarray([t["vector"] for t in finished])
    y = np.asarray([t["score"]["delta_from_baseline_s"] for t in finished])
    variance = np.asarray([t["score"]["paired_mean_variance"] for t in finished])
    gp = GaussianProcess(x, y, variance)
    iteration = sum(t["proposal"]["kind"] == "gp_expected_improvement" for t in trials)
    proposal_seed = seed + 10000 + iteration
    rng = np.random.default_rng(proposal_seed)
    vectors = qmc.Sobol(d=len(space), scramble=True, seed=proposal_seed).random_base2(pool_power)
    complete_indices = [i for i, t in enumerate(finished) if t["score"]["complete"]]
    best_index = min(complete_indices or range(len(finished)), key=lambda i: y[i])
    near = np.clip(x[best_index] + rng.normal(0., .13, (512, len(space))), 0., 1.)
    pool = _unique_pool(space, np.vstack((vectors, near)), seen)
    if len(pool) == 0:
        raise ValueError("No unevaluated parameter setting remains in the acquisition pool")
    mean, sigma = gp.predict(pool)
    incumbent = float(np.min(gp.predict(x[complete_indices] if complete_indices else x)[0]))
    improvement = expected_improvement(mean, sigma, incumbent)
    selected = int(np.argmax(improvement))
    return pool[selected], dict(kind="gp_expected_improvement", index=iteration,
                               random_seed=proposal_seed, candidate_count=len(pool),
                               predicted_delta_s=float(mean[selected]), predicted_std_s=float(sigma[selected]),
                               expected_improvement_s=float(improvement[selected]),
                               fitted_normalized_length_scale=float(gp.length))


def _run_job(module_name, job):
    """Import in the persistent worker; the evaluator enforces the public API."""
    evaluator = importlib.import_module(module_name)
    return evaluator.run_one(job)


def _failure_row(seed, name, spec, error):
    return dict(seed=seed, strategy=name, policy=spec["policy"], label=spec["label"],
                completed=False, source_count=0, cleared=0, per_source_s=None,
                elapsed_per_source_s=0., time_s=0., costs={}, counters={}, phases={},
                error=error, wall_s=0.)


def _persist(state, output):
    finished = [t for t in state["trials"] if t["status"] == "complete"]
    eligible = [t for t in finished if t["score"]["complete"]]
    winner = min(eligible, key=lambda t: t["score"]["objective"]) if eligible else None
    state["best_trial"] = winner["name"] if winner else None
    state["best_mean_per_source_s"] = winner["score"]["objective"] if winner else None
    running_best, trajectory = None, []
    for trial in finished:
        if trial["score"]["complete"] and (running_best is None or trial["score"]["objective"] < running_best["score"]["objective"]):
            running_best = trial
        trajectory.append(dict(trial=trial["name"], objective=trial["score"]["objective"],
                               complete=trial["score"]["complete"],
                               best_trial=running_best["name"] if running_best else None,
                               best_mean_per_source_s=running_best["score"]["objective"] if running_best else None))
    state["best_trajectory"] = trajectory
    write_json(output / "checkpoint.json", state)
    write_json(output / "best_trajectory.json", trajectory)
    if winner:
        write_json(output / "best_config.json", winner["config"])
        write_json(output / "best_parameters.json", winner["config"]["parameters"])


def run_search(*, evaluator_module, base_config, space_spec, seeds, output,
               trials=33, initial=8, workers=6, random_seed=20260913,
               failure_penalty=1e6, pool_power=11, resume=False,
               keep_histories=False, max_new_trials=None):
    """Run/resume complete paired trials using one persistent process pool.

    ``max_new_trials`` intentionally checkpoints a partial run. It is useful for
    interactive compute budgets; the saved manifest still states the full budget.
    Selection data must later be followed by unused validation scenes.
    """
    if workers < 1 or initial < 1 or trials < 1 or pool_power < 1 or failure_penalty <= 0:
        raise ValueError("workers, initial, trials, pool power and failure penalty must be positive")
    if max_new_trials is not None and max_new_trials < 1:
        raise ValueError("max_new_trials must be positive")
    seeds = [int(s) for s in seeds]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Nonempty unique common training seeds are required")
    evaluator = importlib.import_module(evaluator_module)
    baseline = resolve_baseline(evaluator, base_config)
    space = SearchSpace(space_spec)
    baseline_vector = space.encode(baseline["parameters"])
    output = Path(output).resolve()
    settings = dict(evaluator_module=evaluator_module, base_config=baseline,
                    space=space.specification, train_seeds=seeds, initial_sobol=initial,
                    random_seed=random_seed, failure_penalty=failure_penalty,
                    pool_power=pool_power, keep_histories=keep_histories)
    checkpoint = output / "checkpoint.json"
    if resume:
        if not checkpoint.exists():
            raise ValueError("--resume requires an existing checkpoint.json")
        state = read_json(checkpoint)
        if state.get("format_version") != FORMAT_VERSION or state["settings"] != settings:
            raise ValueError("Resume settings differ from saved seeds/space/baseline/evaluator/search configuration")
        check_sources(evaluator, state["code_sha256"], output)
        if trials < len(state["trials"]):
            raise ValueError("Requested total trial budget is smaller than the checkpoint")
        state["requested_trials"] = trials
        state["status"] = "running"
    else:
        if output.exists() and any(output.iterdir()):
            raise ValueError("Use a new output directory, or --resume the unchanged experiment")
        output.mkdir(parents=True, exist_ok=True)
        state = dict(format_version=FORMAT_VERSION,
                     algorithm="normalized_Matern52_GP_expected_improvement_paired_scene_noise",
                     scope="local_synthetic_parameter_selection",
                     started_utc=datetime.now(timezone.utc).isoformat(),
                     settings=settings, requested_trials=trials, status="running", trials=[],
                     code_sha256=source_hashes(evaluator),
                     environment=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                                      platform=platform.platform(), workers=workers,
                                      omp_threads=os.environ.get("OMP_NUM_THREADS"),
                                      openblas_threads=os.environ.get("OPENBLAS_NUM_THREADS")),
                     limitation="Best tested training candidate; no global optimum claim and no held-out evaluation in this run")
        write_json(output / "manifest.json", {k: v for k, v in state.items() if k != "trials"})
        _persist(state, output)
    completed_now = 0
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as executor:
        while len([t for t in state["trials"] if t["status"] == "complete"]) < trials:
            check_sources(evaluator, state["code_sha256"], output)
            pending = [t for t in state["trials"] if t["status"] != "complete"]
            if len(pending) > 1:
                raise ValueError("Checkpoint contains multiple unfinished sequential trials")
            if pending:
                trial = pending[0]
            else:
                index = len(state["trials"])
                if index == 0:
                    vector, proposal = baseline_vector, dict(kind="baseline")
                else:
                    vector, proposal = propose(space, state["trials"], initial, random_seed, pool_power)
                fields = ({d["name"]: baseline["parameters"][d["name"]] for d in space.dimensions}
                          if index == 0 else space.decode(vector))
                specification = dict(baseline, parameters=dict(baseline["parameters"], **fields))
                name = f"trial_{index:03d}"
                configuration = evaluator.resolve_configs({name: specification})[name]
                # Validation/default expansion must not silently alter proposed parameters.
                if not np.allclose(space.encode(configuration["parameters"]), vector, atol=1e-12, rtol=0):
                    raise ValueError("Evaluator altered a proposed search coordinate")
                trial = dict(name=name, index=index, status="running", proposal=proposal,
                             vector=vector.tolist(), searched_parameters=fields, config=configuration, rows=[])
                state["trials"].append(trial)
            directory = output / "trials" / trial["name"]
            write_json(directory / "config.json", trial["config"])
            _persist(state, output)
            known_seeds = {r["seed"] for r in trial["rows"]}
            history_directory = str(directory / "histories") if keep_histories else None
            jobs = [(s, trial["name"], trial["config"], history_directory, False)
                    for s in seeds if s not in known_seeds]
            futures = {executor.submit(_run_job, evaluator_module, job): job[0] for job in jobs}
            for future in as_completed(futures):
                seed = futures[future]
                try:
                    row = future.result()
                    if not isinstance(row, dict) or row.get("seed") != seed:
                        raise ValueError("Evaluator returned an invalid row or wrong seed")
                except Exception:
                    row = _failure_row(seed, trial["name"], trial["config"], traceback.format_exc())
                trial["rows"].append(row)
                trial["rows"].sort(key=lambda r: r["seed"])
                write_json(directory / f"{seed}.summary.json", row)
                _persist(state, output)
            write_json(directory / "rows.json", trial["rows"])
            check_sources(evaluator, state["code_sha256"], output)
            baseline_rows = state["trials"][0]["rows"]
            trial["score"] = summarize(trial["rows"], seeds, failure_penalty, baseline_rows)
            trial["status"] = "complete"
            write_json(directory / "trial.json", trial)
            _persist(state, output)
            completed_now += 1
            print(json.dumps(dict(trial=trial["name"], proposal=trial["proposal"]["kind"],
                                  **trial["score"], best_trial=state["best_trial"],
                                  best_mean_per_source_s=state["best_mean_per_source_s"],
                                  invocation_wall_s=round(time.perf_counter()-started, 3)), ensure_ascii=False), flush=True)
            if max_new_trials is not None and completed_now >= max_new_trials:
                break
    finished = len([t for t in state["trials"] if t["status"] == "complete"])
    state["status"] = "complete" if finished >= trials else "checkpointed"
    state["completed_trials"] = finished
    state["last_finished_utc"] = datetime.now(timezone.utc).isoformat()
    _persist(state, output)
    write_json(output / "optimization.json", state)
    return state


def parse_seeds(value, seed_start, count):
    if value:
        path = Path(value)
        if path.is_file():
            data = read_json(path)
            seeds = data.get("seeds", data.get("train_seeds")) if isinstance(data, dict) else data
        else:
            seeds = [int(s.strip()) for s in value.split(",") if s.strip()]
    else:
        seeds = list(range(seed_start, seed_start + count))
    if not isinstance(seeds, list) or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Training seed list must be nonempty and unique")
    return [int(s) for s in seeds]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-config", required=True, help="JSON with one complete {policy, parameters, label} strategy")
    parser.add_argument("--space", required=True, help="JSON of named float/int low/high bounds")
    parser.add_argument("--evaluator-module", default="question4.free_joint.evaluate")
    parser.add_argument("--train-seeds", help="JSON seed-list path or comma-separated seeds; overrides start/count")
    parser.add_argument("--seed-start", type=int, default=20294000)
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--trials", type=int, default=33, help="Total: baseline + Sobol + GP/EI; default 1+8+24")
    parser.add_argument("--initial", type=int, default=8, help="Unique Sobol proposals after the baseline")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--random-seed", type=int, default=20260913)
    parser.add_argument("--failure-penalty", type=float, default=1e6)
    parser.add_argument("--pool-power", type=int, default=11, help="Acquisition pool has 2**power Sobol +512 local samples")
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--keep-histories", action="store_true")
    parser.add_argument("--max-new-trials", type=int, help="Checkpoint after this many additional trials, then resume later")
    parser.add_argument("--allow-short-run", action="store_true", help="Explicit smoke/limited-budget exception to 8 Sobol +24 EI")
    args = parser.parse_args()
    if not args.allow_short_run and (args.initial < 8 or args.trials < 1 + args.initial + 24):
        parser.error("A regular search requires >=8 Sobol and >=24 EI proposals; use --allow-short-run for a labelled smoke run")
    run_search(evaluator_module=args.evaluator_module,
               base_config=read_json(args.base_config), space_spec=read_json(args.space),
               seeds=parse_seeds(args.train_seeds, args.seed_start, args.count), output=args.output,
               trials=args.trials, initial=args.initial, workers=args.workers, random_seed=args.random_seed,
               failure_penalty=args.failure_penalty, pool_power=args.pool_power,
               resume=args.resume, keep_histories=args.keep_histories, max_new_trials=args.max_new_trials)


if __name__ == "__main__":
    main()

"""Offline, shareable figures from recorded Q4 experiments.

No planner or simulator is imported. Curves use actual completed training trials
and paired validation rows. Trajectories use recorded commands; emitter truth is
read only after selecting bad/good cases by measured mission cost, for display.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
import numpy as np


BASE = Path(__file__).resolve().parent
LABELS = {"old_joint": "最初联合方案", "previous_dynamic": "此前动态方案",
          "free_joint": "上一最佳：自由补点", "tuned_joint": "本轮优化方案"}
COLORS = {"old_joint": "#8297ae", "previous_dynamic": "#ba964d",
          "free_joint": "#398981", "tuned_joint": "#315ea7"}
COSTS = [("movement", "移动", "#698ab2"), ("measure", "测量", "#d5a054"),
         ("switch", "切频", "#a383bd"), ("clear_success", "成功清除", "#4c9b80"),
         ("clear_failure", "失败清除", "#ce6b75")]


def configure():
    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
                         "axes.unicode_minus": False, "font.size": 10,
                         "axes.titleweight": "bold", "text.color": "#20384c",
                         "axes.labelcolor": "#3b576d", "axes.edgecolor": "#cad4df",
                         "xtick.color": "#587184", "ytick.color": "#587184",
                         "figure.facecolor": "#f4f7fa", "axes.facecolor": "#ffffff",
                         "savefig.facecolor": "#f4f7fa"})


def read(path):
    path = Path(path)
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            return json.load(stream)
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def save(fig, path):
    fig.savefig(path, dpi=165, bbox_inches="tight", pad_inches=.16)
    plt.close(fig)
    return str(Path(path).resolve())


def complete(row):
    return (bool(row.get("completed")) and row.get("per_source_s") is not None
            and np.isfinite(row["per_source_s"]) and row.get("source_count", 0) > 0
            and row.get("cleared", row["source_count"]) == row["source_count"])


def strategy_order(data):
    available = list(dict.fromkeys(row["strategy"] for row in data["rows"]))
    return [name for name in ("old_joint", "previous_dynamic", "free_joint", "tuned_joint") if name in available]


def paired_stats(rows, current, reference):
    first = {r["seed"]: r for r in rows if r["strategy"] == current}
    second = {r["seed"]: r for r in rows if r["strategy"] == reference}
    shared = sorted(first.keys() & second.keys())
    eligible = [s for s in shared if complete(first[s]) and complete(second[s])]
    delta = np.asarray([first[s]["per_source_s"]-second[s]["per_source_s"] for s in eligible])
    all_complete = len(eligible) == len(shared) and set(first) == set(second)
    result = dict(current=current, reference=reference, direction="current minus reference, seconds/source",
                  paired_count=len(shared), completed_pairs=len(eligible),
                  excluded_pairs=len(shared)-len(eligible), complete_group=all_complete,
                  unmatched_seed_count=len(set(first)^set(second)), seeds=eligible,
                  differences_s=delta.tolist(), mean_difference_s=None, bootstrap95_s=None,
                  completed_subset_mean_s=float(np.mean(delta)) if len(delta) else None,
                  faster_count=int(np.sum(delta < 0)), bootstrap_draws=5000, bootstrap_seed=419)
    if len(delta) and all_complete:
        rng = np.random.default_rng(419)
        boot = delta[rng.integers(0, len(delta), size=(5000, len(delta)))].mean(axis=1)
        result.update(mean_difference_s=float(delta.mean()), bootstrap95_s=np.quantile(boot, [.025, .975]).tolist())
    return result


def group_statistics(data, names):
    statistics = {}
    for name in names:
        rows = [r for r in data["rows"] if r["strategy"] == name]
        successful = [r for r in rows if complete(r)]
        all_complete = len(rows) == len(successful)
        values = np.asarray([r["per_source_s"] for r in successful])
        statistics[name] = dict(count=len(rows), completed_count=len(successful),
                                source_count=sum(r["source_count"] for r in rows),
                                cleared=sum(r.get("cleared", 0) for r in rows),
                                mean_per_source_s=float(values.mean()) if all_complete else None,
                                p90_per_source_s=float(np.quantile(values, .9)) if all_complete else None,
                                costs_per_source={k: float(np.mean([r["costs"].get(k, 0.)/r["source_count"]
                                                                   for r in rows])) for k, _, _ in COSTS},
                                values=values.tolist(), failed_seeds=[r["seed"] for r in rows if not complete(r)])
    return statistics


def bayes_figure(checkpoint, output, compatibility=False):
    data = read(checkpoint)
    trials = data.get("trials", [])
    if not trials:
        return None
    finished = [t for t in trials if t.get("status") == "complete"]
    good = [t for t in finished if t.get("score", {}).get("complete")]
    if not good:
        return None
    fig, ax = plt.subplots(figsize=(13.5, 6.2))
    kinds = [("baseline", "未调参基准", "#667f95", "D"),
             ("sobol", "Sobol 初始探索", "#438e99", "o"),
             ("gp_expected_improvement", "GP＋EI 自适应提议", "#c48e3b", "^")]
    for kind, label, color, marker in kinds:
        subset = [t for t in good if t["proposal"]["kind"] == kind]
        if subset:
            ax.scatter([t["index"]+1 for t in subset], [t["score"]["mean_per_source_s"] for t in subset],
                       s=63, marker=marker, color=color, edgecolors="white", linewidths=.6,
                       label=f"{label}（{len(subset)}组）", zorder=4)
    best = np.inf
    xx, yy = [], []
    for trial in finished:
        if trial["score"].get("complete"):
            best = min(best, trial["score"]["mean_per_source_s"])
        if np.isfinite(best):
            xx.append(trial["index"]+1)
            yy.append(float(best))
    ax.step(xx, yy, where="post", color="#2c745f", lw=2, label="截至该组的训练最优", zorder=3)
    initial = data.get("settings", {}).get("initial_sobol", 8)
    ax.axvline(initial+1.5, color="#a2adba", ls="--", lw=1.1)
    ax.text(initial+1.6, 1.018, "开始使用 GP＋EI", transform=ax.get_xaxis_transform(), va="bottom", fontsize=9)
    failures = [t for t in finished if not t["score"].get("complete")]
    if failures:
        ax.scatter([t["index"]+1 for t in failures], [.97]*len(failures), transform=ax.get_xaxis_transform(),
                   marker="x", color="#c15762", s=65, label="未全清：不参与最优选择", zorder=5)
    final = min(good, key=lambda t: t["score"]["mean_per_source_s"])
    alignment = "right" if final["index"]+1 > .8*max(t["index"]+1 for t in trials) else "center"
    ax.annotate(f"训练最优 {final['score']['mean_per_source_s']:.2f} 秒/源",
                (final["index"]+1, final["score"]["mean_per_source_s"]),
                xytext=(0, -24), textcoords="offset points", ha=alignment, fontsize=10,
                color="#245c4d", bbox=dict(boxstyle="round,pad=.3", fc="white", ec="#cbded5"))
    ax.set_xlim(.4, max(t["index"]+1 for t in trials)+.7)
    ax.margins(y=.19)
    ax.set_xlabel("实际评估顺序（第 1 组为基准，每组使用相同训练场景）")
    ax.set_ylabel("全清训练场景：完整平均秒/源")
    ax.grid(axis="y", alpha=.2)
    ax.legend(frameon=False, ncol=2, loc="upper right", fontsize=9)
    title = "贝叶斯参数优化：实际成绩与累积最好"
    if compatibility:
        title = "兼容绘图测试 · " + title
    fig.suptitle(title, x=.07, ha="left", fontsize=18, y=.985)
    seeds = data.get("settings", {}).get("train_seeds", [])
    subtitle = (f"{len(finished)} 组已完成评估；每组 {len(seeds)} 个公共训练场景。"
                "未全清试验受到完整失败惩罚，图中不把其已耗时间当作任务成绩。")
    fig.text(.07, .915, subtitle, fontsize=9, color="#63768a")
    fig.text(.07, .025, "本图用于参数选择，不能代替独立验证；有限预算中的最好配置不等于全局最优。", fontsize=9, color="#63768a")
    fig.subplots_adjust(left=.09, right=.98, top=.85, bottom=.15)
    return dict(file=save(fig, output / "bayes_progress.png"), trial_count=len(finished),
                best_trial=final["name"], best_training_mean_s=final["score"]["mean_per_source_s"],
                failed_trial_count=len(failures), checkpoint=str(Path(checkpoint).resolve()))


def validation_figure(data, names, current, output, compatibility=False):
    summary = group_statistics(data, names)
    references = [n for n in names if n != current]
    comparisons = [paired_stats(data["rows"], current, n) for n in references]
    fig = plt.figure(figsize=(15.6, 10.1))
    grid = fig.add_gridspec(2, 2, height_ratios=[1., 1.1])
    ax = fig.add_subplot(grid[0, 0])
    left = np.zeros(len(names))
    for key, label, color in COSTS:
        values = np.asarray([summary[n]["costs_per_source"][key] for n in names])
        ax.barh(np.arange(len(names)), values, left=left, height=.52, color=color, label=label)
        left += values
    for i, name in enumerate(names):
        s = summary[name]
        value = f"{s['mean_per_source_s']:.2f}" if s["mean_per_source_s"] is not None else "未全清"
        ax.text(left[i]+max(left)*.02, i, value, va="center", fontsize=10, weight="bold")
    ax.set_yticks(range(len(names)), [LABELS.get(n, n) for n in names])
    ax.invert_yaxis()
    ax.set_xlim(0, max(left)*1.18)
    ax.set_xlabel("平均完整时间及其费用分解（秒/源）")
    ax.set_title("所有动作都计费", loc="left", pad=15)
    ax.grid(axis="x", alpha=.16)
    ax.legend(loc="upper center", bbox_to_anchor=(.52, -.22), ncol=3, frameon=False, fontsize=9)
    ax = fig.add_subplot(grid[0, 1])
    for i, name in enumerate(names):
        values = np.asarray(summary[name]["values"])
        jitter = np.random.default_rng(610+i).uniform(-.12, .12, len(values))
        ax.scatter(i+jitter, values, s=18, alpha=.4, color=COLORS.get(name, "#8196ad"))
        if len(values):
            ax.boxplot([values], positions=[i], widths=.42, showfliers=False,
                       medianprops=dict(color="#1e3e57", linewidth=2),
                       boxprops=dict(color="#7890a5"), whiskerprops=dict(color="#7890a5"),
                       capprops=dict(color="#7890a5"))
        s = summary[name]
        ax.text(i, 1.025, f"全清 {s['completed_count']}/{s['count']}",
                transform=ax.get_xaxis_transform(), ha="center", fontsize=9)
    ax.axhline(500, color="#c46563", lw=1.2, ls=":")
    ax.set_xticks(range(len(names)), [LABELS.get(n, n).replace("：", "\n") for n in names])
    ax.set_ylabel("每场完整时间/源（秒）")
    ax.set_title("逐场分布与 500 秒目标线", loc="left", pad=25)
    ax.grid(axis="y", alpha=.16)
    for column in range(2):
        ax = fig.add_subplot(grid[1, column])
        if column >= len(comparisons):
            ax.axis("off")
            continue
        comparison = comparisons[column]
        reference = comparison["reference"]
        by_current = {r["seed"]: r for r in data["rows"] if r["strategy"] == current}
        by_reference = {r["seed"]: r for r in data["rows"] if r["strategy"] == reference}
        x = np.asarray([by_reference[s]["per_source_s"] for s in comparison["seeds"]])
        y = np.asarray([by_current[s]["per_source_s"] for s in comparison["seeds"]])
        if len(x):
            bounds = [float(min(x.min(), y.min())-20), float(max(x.max(), y.max())+20)]
            ax.plot(bounds, bounds, "--", color="#a0adba", lw=1.1)
            ax.scatter(x, y, s=24, color=np.where(y <= x, "#3b927c", "#c3767d"), alpha=.8)
            ax.set_xlim(bounds)
            ax.set_ylim(bounds)
        ax.set_xlabel(f"{LABELS.get(reference, reference)}（秒/源）")
        ax.set_ylabel(f"{LABELS.get(current, current)}（秒/源）")
        ax.set_title(f"同场配对：相对{LABELS.get(reference, reference)}", loc="left", pad=12)
        if comparison["mean_difference_s"] is not None:
            lo, hi = comparison["bootstrap95_s"]
            note = (f"平均差 {comparison['mean_difference_s']:+.2f} 秒/源\n"
                    f"配对 bootstrap 95%：[{lo:+.2f}, {hi:+.2f}]\n"
                    f"更快 {comparison['faster_count']}/{comparison['paired_count']} 场；下方表示本轮更快")
        else:
            note = f"仅显示 {comparison['completed_pairs']} 个完成配对；存在缺失/失败，不给完整组平均差"
        ax.text(.035, .96, note, transform=ax.transAxes, va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=.4", fc="white", ec="#dfe6ec", alpha=.95))
        ax.grid(alpha=.13)
    title = "兼容绘图测试：旧验证数据" if compatibility else "第四问独立验证：完整费用与共同场景配对"
    fig.suptitle(title, fontsize=19, ha="left", x=.035, y=.985)
    source_total = summary[names[0]]["source_count"]
    fig.text(.035, .94, f"本地合成；每策略 {summary[names[0]]['count']} 场、{source_total} 个源。每场先算总时间/源数，再等权平均。",
             fontsize=10, color="#63788b")
    fig.text(.035, .02, "费用包含移动、全部测量、切频、成功与失败清除。散点只画完成任务；若有失败，不给完整组平均成绩。", fontsize=9, color="#63788b")
    fig.subplots_adjust(left=.13, right=.98, top=.86, bottom=.09, hspace=.65, wspace=.36)
    return dict(file=save(fig, output / "validation_overview.png"), summary=summary, paired=comparisons)


def difference_figure(comparisons, output):
    if not comparisons:
        return None
    fig, axes = plt.subplots(1, len(comparisons), figsize=(7.1*len(comparisons), 5.7), squeeze=False)
    for ax, comparison in zip(axes[0], comparisons):
        delta = np.asarray(comparison["differences_s"])
        order = np.argsort(delta)
        ax.bar(np.arange(len(delta)), delta[order], width=.9,
               color=np.where(delta[order] <= 0, "#4d977f", "#c77580"))
        ax.axhline(0, color="#72889d", lw=1)
        if comparison["mean_difference_s"] is not None:
            ax.axhline(comparison["mean_difference_s"], color="#3d5874", lw=1.2, ls="--")
        if len(delta):
            for index in dict.fromkeys([0, len(delta)-1]):
                source_index = order[index]
                ax.annotate(str(comparison["seeds"][source_index]), (index, delta[source_index]),
                            xytext=(0, 9 if delta[source_index] >= 0 else -16),
                            textcoords="offset points", ha="center", fontsize=8)
        ax.margins(y=.28, x=.02)
        ax.set_xlabel("相同种子的配对差，按差值升序排列")
        ax.set_ylabel("本轮 − 对照（秒/源）")
        ax.set_title(f"相对{LABELS.get(comparison['reference'], comparison['reference'])}", loc="left", pad=15)
        ax.grid(axis="y", alpha=.16)
    fig.suptitle("逐场收益与退步：负值表示本轮更快", ha="left", x=.065, y=.97, fontsize=17)
    fig.text(.065, .025, "这是同场景比较，不跨不同种子组相减；柱形图保留全部完成配对，包括表现变差的样本。", fontsize=9, color="#63788b")
    fig.subplots_adjust(left=.075, right=.98, top=.82, bottom=.16, wspace=.29)
    return save(fig, output / "paired_differences.png")


def choose_cases(data, current, reference, count=3):
    current_rows = {r["seed"]: r for r in data["rows"] if r["strategy"] == current and complete(r)}
    reference_rows = {r["seed"]: r for r in data["rows"] if r["strategy"] == reference and complete(r)}
    if not current_rows:
        return []
    chosen = {}
    def add(seed, tag):
        chosen.setdefault(seed, []).append(tag)
    add(max(current_rows, key=lambda s: current_rows[s]["per_source_s"]), "本轮每源耗时最长")
    shared = set(current_rows) & set(reference_rows)
    if shared:
        delta = lambda s: current_rows[s]["per_source_s"]-reference_rows[s]["per_source_s"]
        add(max(shared, key=delta), f"相对{LABELS.get(reference, reference)}退步最多")
        add(min(shared, key=delta), f"相对{LABELS.get(reference, reference)}进步最多")
    for seed in sorted(current_rows, key=lambda s: current_rows[s]["per_source_s"], reverse=True):
        if len(chosen) >= count:
            break
        if seed not in chosen:
            add(seed, "补充慢案例（极值案例有重合）")
    return [dict(seed=seed, reasons=tags, current=current, reference=reference,
                 current_per_source_s=current_rows[seed]["per_source_s"],
                 delta_from_reference_s=(current_rows[seed]["per_source_s"]-reference_rows[seed]["per_source_s"])
                 if seed in reference_rows else None)
            for seed, tags in chosen.items()][:count]


def locate_history(filename, validation, output=None, seed=None, strategy=None):
    from .report import locate_history as portable_history
    return portable_history(filename, validation, output, seed, strategy)


def motion_data(history, expected):
    commands = history.get("commands") or []
    if not commands:
        raise ValueError("Cannot draw a route without actual recorded commands")
    if abs(float(history["summary"]["time_s"])-float(expected["time_s"])) > 1e-6:
        raise AssertionError("Selected history total differs from validation row")
    if (history["summary"].get("seed") != expected["seed"]
            or history["summary"].get("strategy") != expected["strategy"]):
        raise AssertionError("Selected history scenario/strategy differs from validation row")
    position = np.zeros(2)
    segments, movements, hit, miss, cleared, failed = [], [], [], [], [], []
    total_distance = 0.
    for i, row in enumerate(commands):
        q = np.asarray(row["position"], dtype=float)
        start = np.asarray(row.get("from", position), dtype=float)
        distance = float(np.linalg.norm(q-start))
        if abs(distance-float(row["move_s"])*5.) > 1e-3:
            raise AssertionError(f"Recorded move geometry disagrees with charged distance at command {i}")
        if distance > 1e-6:
            segments.append((start, q))
            movements.append(dict(position=q, command=i+1, time_s=row["time_s"]))
            total_distance += distance
        response = row.get("response", {})
        if row["kind"] == "measure":
            (hit if response.get("measure_result") in ("direction", "near") else miss).append(q)
        elif row["kind"] == "clear":
            (cleared if response.get("clear_result") == "success" else failed).append(q)
        position = q
    counts = Counter(row["kind"] for row in commands)
    last_clear = max((row["time_s"] for row in commands if row["kind"] == "clear"
                      and row.get("response", {}).get("clear_result") == "success"), default=0.)
    return dict(segments=segments, movements=movements, hit=hit, miss=miss, cleared=cleared,
                failed=failed, end=position, metrics=dict(command_count=len(commands),
                movement_m=total_distance, physical_moves=len(movements), measurements=counts["measure"],
                clear_success=len(cleared), clear_failure=len(failed),
                search_after_last_clear_s=float(expected["time_s"])-float(last_clear)))


def _scatter_points(ax, points, **kwargs):
    if points:
        p = np.asarray(points)
        ax.scatter(p[:, 0], p[:, 1], **kwargs)


def case_figure(case, data, names, validation, output, compatibility=False):
    rows = {r["strategy"]: r for r in data["rows"] if r["seed"] == case["seed"]}
    histories, paths, motions = {}, {}, {}
    for name in names:
        if name not in rows:
            raise ValueError(f"Selected seed {case['seed']} missing strategy {name}")
        path = locate_history(rows[name]["history_file"], validation, output.parent, case["seed"], name)
        histories[name] = read(path)
        paths[name] = str(path.resolve())
        motions[name] = motion_data(histories[name], rows[name])
    truth = histories[names[0]].get("truth", [])
    # Truth is accessed only here, after cost-based sample selection.
    truth_key = lambda sources: sorted((s["channel"], s["x"], s["y"], s["directional"], s["orientation"]) for s in sources)
    if any(truth_key(h.get("truth", [])) != truth_key(truth) for h in histories.values()):
        raise AssertionError("Paired histories do not describe the same emitter scene")
    all_points = [np.asarray(row["position"]) for h in histories.values() for row in h["commands"]]
    extent = max(2100., max(float(np.abs(p).max()) for p in all_points)+100.)
    extent = float(np.ceil(extent/100.)*100.)
    fig, axes = plt.subplots(1, len(names), figsize=(6.05*len(names), 7.55), squeeze=False)
    for ax, name in zip(axes[0], names):
        motion, row = motions[name], rows[name]
        ax.add_patch(Circle((0, 0), 1800, fill=False, ec="#a2afbd", ls="--", lw=1.2, zorder=1))
        for start, end in motion["segments"]:
            ax.plot([start[0], end[0]], [start[1], end[1]], color="#728fac", lw=1.25, alpha=.82, zorder=2)
        segments = motion["segments"]
        long_segments = [segment for segment in segments if np.linalg.norm(segment[1]-segment[0]) >= 200.]
        for index in np.linspace(0, len(long_segments)-1, min(14, len(long_segments)), dtype=int):
            start, end = long_segments[index]
            tail = start+.48*(end-start)
            head = start+.64*(end-start)
            ax.annotate("", head, tail, arrowprops=dict(arrowstyle="->", color="#516e91", lw=1.25), zorder=3)
        _scatter_points(ax, motion["miss"], s=15, facecolors="none", edgecolors="#9ba9b7", linewidths=.65, zorder=3)
        _scatter_points(ax, motion["hit"], s=17, color="#346fa0", edgecolors="white", linewidths=.35, zorder=4)
        _scatter_points(ax, motion["failed"], s=32, color="#c76470", marker="x", linewidths=1.1, zorder=5)
        _scatter_points(ax, motion["cleared"], s=63, color="#348566", marker="*", edgecolors="white", linewidths=.45, zorder=6)
        for source in truth:
            x, y = source["x"], source["y"]
            ax.scatter(x, y, s=31, marker="D", facecolors="none", edgecolors="#354553", linewidths=1., zorder=7)
            ax.annotate(f"c{source['channel']}", (x, y), xytext=(5, 5), textcoords="offset points", fontsize=7.4,
                        bbox=dict(boxstyle="round,pad=.10", fc="white", ec="none", alpha=.75), zorder=9)
            if source["directional"]:
                theta = source["orientation"]
                end = (x+190*np.cos(theta), y+190*np.sin(theta))
                ax.annotate("", end, (x, y), arrowprops=dict(arrowstyle="-|>", color="#354553", lw=1.2), zorder=8)
        ax.scatter(0, 0, marker="s", s=46, color="#1c3951", edgecolors="white", linewidths=.7, zorder=10)
        ax.scatter(*motion["end"], marker="P", s=45, color="#a17d2d", edgecolors="white", linewidths=.6, zorder=10)
        moves = motion["movements"]
        if moves:
            for index in np.linspace(0, len(moves)-1, min(9, len(moves)), dtype=int):
                item = moves[index]
                ax.annotate(f"a{item['command']}", item["position"], xytext=(-4, -10), textcoords="offset points",
                            fontsize=6.6, color="#7a4e28", ha="right", zorder=9,
                            bbox=dict(boxstyle="round,pad=.08", fc="white", ec="none", alpha=.7))
        metrics = motion["metrics"]
        ax.set_title(f"{LABELS.get(name, name)}\n{row['per_source_s']:.2f} 秒/源 · 共 {row['time_s']:.0f} 秒",
                     loc="left", fontsize=12, pad=14)
        ax.text(0., -.115,
                f"移动 {metrics['movement_m']/1000:.2f} km  |  测量 {metrics['measurements']} 次\n"
                f"成功/失败清除 {metrics['clear_success']}/{metrics['clear_failure']}  |  最后清除后 {metrics['search_after_last_clear_s']:.0f} 秒",
                transform=ax.transAxes, va="top", fontsize=9, color="#597286", linespacing=1.65)
        ax.set_xlim(-extent, extent)
        ax.set_ylim(-extent, extent)
        ax.set_aspect("equal")
        ax.set_xlabel("x（米）", labelpad=2)
        ax.set_ylabel("y（米）", labelpad=2)
        ax.grid(alpha=.12)
        ax.set_xticks(np.arange(-2000, 2001, 1000))
        ax.set_yticks(np.arange(-2000, 2001, 1000))
    title = f"场景 {case['seed']}：{'；'.join(case['reasons'])}"
    if compatibility:
        title = "兼容绘图测试 · " + title
    fig.suptitle(title, x=.035, ha="left", fontsize=17, y=.99)
    handles = [Line2D([], [], color="#728fac", lw=1.4, label="实际移动（小箭头示方向）"),
               Line2D([], [], marker="o", color="none", markerfacecolor="#346fa0", markeredgecolor="white", markersize=6, label="有信号测量"),
               Line2D([], [], marker="o", color="none", markerfacecolor="none", markeredgecolor="#9ba9b7", markersize=5, label="无信号测量"),
               Line2D([], [], marker="*", color="none", markerfacecolor="#348566", markersize=10, label="成功清除"),
               Line2D([], [], marker="x", color="#c76470", linestyle="none", markersize=6, label="失败清除"),
               Line2D([], [], marker="D", color="none", markeredgecolor="#354553", markersize=5, label="真实源；箭头为发射朝向"),
               Line2D([], [], marker="s", color="#1c3951", linestyle="none", markersize=5, label="原点"),
               Line2D([], [], marker="P", color="#a17d2d", linestyle="none", markersize=6, label="终点")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.51, .045), ncol=4, frameon=False,
               fontsize=9, columnspacing=1.8, handletextpad=.6)
    fig.text(.035, .013, "只连接实际移动；同点多次扫描重叠显示。a 编号为真实动作序号，抽取少量标注。源位置和朝向仅用于事后显示，未输入策略。",
             fontsize=9, color="#617789")
    fig.subplots_adjust(left=.045, right=.985, top=.85, bottom=.23, wspace=.19)
    path = output / f"case_{case['seed']}.png"
    return dict(**case, figure=save(fig, path), histories=paths,
                motion_metrics={name: motions[name]["metrics"] for name in names})


def generate(output, validation=None, checkpoint=None, current="tuned_joint", reference="free_joint", compatibility=False):
    configure()
    output = Path(output).resolve()
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    validation = Path(validation) if validation else output / "validate.json"
    checkpoint = Path(checkpoint) if checkpoint else output / "bayes" / "checkpoint.json"
    result = dict(scope="offline figures from actual full-mission records", compatibility_test=compatibility,
                  validation=str(validation.resolve()), checkpoint=str(checkpoint.resolve()),
                  bayes=None, validation_figures=None, cases=[], missing_inputs=[])
    if checkpoint.is_file():
        result["bayes"] = bayes_figure(checkpoint, figures, compatibility)
    else:
        result["missing_inputs"].append("bayes/checkpoint.json not present; no optimization curve generated")
    if validation.is_file():
        data = read(validation)
        names = strategy_order(data)
        if current not in names or reference not in names:
            raise ValueError(f"Required strategies {current}/{reference} not present: {names}; use explicit compatibility arguments for older files")
        if len(names) > 3:
            names = [n for n in names if n in ("old_joint", reference, current)]
        overview = validation_figure(data, names, current, figures, compatibility)
        overview["differences_file"] = difference_figure(overview["paired"], figures)
        result["validation_figures"] = overview
        selected = choose_cases(data, current, reference)
        # Record the selection before reading truth/history, including if an input
        # history is missing. No fallback simulation or trajectory fabrication.
        write(figures / "case_selection.json", dict(rule="actual complete-mission elapsed cost only; truth not used", cases=selected))
        result["cases"] = [case_figure(case, data, names, validation, figures, compatibility) for case in selected]
        write(figures / "case_selection.json", dict(rule="actual complete-mission elapsed cost only; truth used offline for source arrows", cases=result["cases"]))
    else:
        result["missing_inputs"].append("validate.json not present; no validation/case figures generated")
    write(figures / "figure_manifest.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=BASE / "outputs", help="Experiment root; PNG files are written to its figures directory")
    parser.add_argument("--validation", type=Path, help="Override the input validation JSON (read only)")
    parser.add_argument("--checkpoint", type=Path, help="Override the input Bayes checkpoint (read only)")
    parser.add_argument("--current", default="tuned_joint")
    parser.add_argument("--reference", default="free_joint")
    parser.add_argument("--compatibility-test", action="store_true", help="Mark every validation figure as an older-data rendering test")
    args = parser.parse_args()
    result = generate(args.output, args.validation, args.checkpoint, args.current, args.reference, args.compatibility_test)
    print(json.dumps(dict(output=str((args.output / 'figures').resolve()),
                          bayes_plot=result["bayes"], cases=[dict(seed=c["seed"], figure=c["figure"]) for c in result["cases"]],
                          missing_inputs=result["missing_inputs"]), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

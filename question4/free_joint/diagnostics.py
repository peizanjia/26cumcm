"""Offline statistics and motion-history figures for dynamic Q4 evaluation.

Only recorded commands determine trajectories and counters. Source truth is
used for figure annotations, never for policy decisions or reconstructed moves.
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
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Wedge
import numpy as np


BASE = Path(__file__).resolve().parent
SHORT = {"old_joint": "旧联合", "previous_dynamic": "上一版动态", "dynamic": "仅动态未知域", "free_joint": "自由补点"}
PALETTE = {"old_joint": "#7294aa", "previous_dynamic": "#bd995b", "dynamic": "#bd995b", "free_joint": "#408d79"}
COSTS = [("movement", "移动", "#5887ad"), ("measure", "测量", "#d6a052"),
         ("switch", "切频", "#9c7db8"), ("clear_success", "清除成功", "#4c9b7b"),
         ("clear_failure", "清除失败", "#c8626f")]
KINDS = {"search": ("探索停点", "#327fb4", "o"),
         "service": ("定位／清除停点", "#349078", "o"),
         "intermediate": ("实际中途停点", "#d09332", "s")}


def configure():
    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
                         "axes.unicode_minus": False, "font.size": 10,
                         "axes.titleweight": "bold", "axes.labelcolor": "#4e6576",
                         "text.color": "#193447", "axes.edgecolor": "#c3d0d9",
                         "xtick.color": "#627a8c", "ytick.color": "#627a8c",
                         "figure.facecolor": "#f3f7fa", "axes.facecolor": "#fcfdff",
                         "savefig.facecolor": "#f3f7fa"})


def read(path):
    path = Path(path)
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            return json.load(stream)
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def answer(row):
    feedback = row["response"]
    return feedback.get("measure_result", feedback.get("clear_result", ""))


def commands(history):
    return history.get("commands") or history.get("frames") or []


def physical_stops(history):
    """Collapse same-place channel scans; retain arrival action and all fees."""
    groups = []
    for row in commands(history):
        q = np.asarray(row["position"], dtype=float)
        if not groups or np.linalg.norm(q-groups[-1]["position"]) > 1e-5:
            groups.append(dict(position=q, rows=[], arrival=row,
                               number=len(groups), move_m=float(row["move_s"])*5.0))
        groups[-1]["rows"].append(row)
    for stop in groups:
        rows = stop["rows"]
        exploratory = any(r["phase"] in ("origin", "search", "side_search") for r in rows)
        servicing = any(r["phase"] in ("service", "side_known") or r["kind"] == "clear" for r in rows)
        reason = stop["arrival"].get("reason", "")
        stop["kind"] = ("intermediate" if "intermediate" in reason else
                        "search" if stop["arrival"]["phase"] in ("origin", "search", "side_search") else "service")
        known_channels = {r["channel"] for r in rows if r["phase"] in ("service", "side_known")}
        stop["shared"] = bool((exploratory and servicing) or len(known_channels) >= 2)
        stop["successful_clear"] = any(r["kind"] == "clear" and answer(r) == "success" for r in rows)
        stop["failed_clear"] = any(r["kind"] == "clear" and answer(r) != "success" for r in rows)
        stop["end_time_s"] = float(rows[-1]["time_s"])
        stop["measurements"] = sum(r["kind"] == "measure" for r in rows)
    return groups


def motion_metrics(history):
    rows = commands(history)
    stops = physical_stops(history)
    found, cleared = {}, {}
    known_negative = 0
    for row in rows:
        c = int(row["channel"])
        value = answer(row)
        known_negative += int(value == "no_signal" and c in found)
        if value in ("near", "direction"):
            found.setdefault(c, row["time_s"])
        if row["kind"] == "clear" and value == "success":
            cleared[c] = row["time_s"]
    count = Counter(s["kind"] for s in stops[1:])
    return dict(physical_stops=max(0, len(stops)-1), stop_kinds=dict(count),
                shared_stops=sum(s["shared"] for s in stops[1:]),
                intermediate_stops=sum(s["kind"] == "intermediate" for s in stops[1:]),
                movement_m=sum(float(r["move_s"])*5.0 for r in rows),
                known_no_signal=known_negative,
                clear_failures=sum(r["kind"] == "clear" and answer(r) != "success" for r in rows),
                clear_successes=len(cleared), discovered=len(found),
                search_after_last_clear_s=(float(rows[-1]["time_s"])-max(cleared.values())) if rows and cleared else None,
                arrival_reasons=dict(Counter(s["arrival"].get("reason", "") for s in stops[1:])),
                interpretations=["共享停点按实际动作判断：同点搜索与服务，或服务至少两个已知频道",
                                 "中途停点按实际到达动作的intermediate原因判断，不把经过线段当测量",
                                 "全局图的a编号为真实动作序号（从1起），为清晰只抽取部分编号"])


def overview(data, output, smoke=False):
    rows = data["rows"]
    summaries = {r["strategy"]: r for r in data["summary"]}
    names = [n for n in SHORT if n in summaries]
    names += [n for n in summaries if n not in names]
    fig, axes = plt.subplots(2, 2, figsize=(16.5, 10.5))
    ax = axes[0, 0]
    left = np.zeros(len(names))
    for key, label, color in COSTS:
        value = np.array([summaries[n]["costs_per_source"].get(key, 0.) for n in names])
        ax.barh(range(len(names)), value, left=left, color=color, height=.48, label=label)
        left += value
    for i, n in enumerate(names):
        label = f"{left[i]:.2f}" if summaries[n].get("failed_count", 0) == 0 else "含未完成费用"
        ax.text(left[i]+max(left)*.015, i, label, va="center", fontsize=10, weight="bold")
    ax.set_yticks(range(len(names)), [SHORT.get(n, n) for n in names])
    ax.set_ylim(len(names)-.25, -.6)
    ax.set_xlim(0, max(left, default=1)*1.2)
    ax.set_xlabel("各场完整总时间／实际源数，再等权平均（s／源）")
    ax.set_title("所有真实动作计费：移动、测量与清除", loc="left", pad=14)
    ax.legend(ncol=3, frameon=False, fontsize=9, loc="lower left")
    ax = axes[0, 1]
    for i, n in enumerate(names):
        values = [r["per_source_s"] for r in rows if r["strategy"] == n and r.get("completed")]
        if values:
            jitter = np.random.default_rng(i+947).uniform(-.12, .12, len(values))
            ax.scatter(i+jitter, values, s=15, alpha=.42, color=PALETTE.get(n, "#7e8c95"))
            ax.boxplot([values], positions=[i], widths=.40, showfliers=False,
                       medianprops=dict(color="#1e3d53", linewidth=2), boxprops=dict(color="#617e94"),
                       whiskerprops=dict(color="#617e94"), capprops=dict(color="#617e94"))
        failed = summaries[n].get("failed_count", 0)
        if failed:
            ax.text(i, 1.01, f"另有{failed}场未完成", transform=ax.get_xaxis_transform(), ha="center", color="#b35361")
    ax.set_xticks(range(len(names)), [SHORT.get(n, n) for n in names])
    ax.axhline(500, color="#c65b56", ls=":", lw=1.4, label="500秒目标")
    ax.set_ylabel("完整任务总时间／源（s）")
    ax.set_title("逐场分布：观察均值以外的长尾", loc="left", pad=14)
    ax.grid(axis="y", alpha=.18)
    ax = axes[1, 0]
    paired = paired_rows(rows)
    if paired:
        values = np.array([a["per_source_s"]-b["per_source_s"] for _, a, b in paired])
        order = np.argsort(values)
        ax.bar(np.arange(len(order)), values[order], width=.9,
               color=np.where(values[order] <= 0, "#4a947e", "#c06b73"))
        ax.axhline(0, color="#698397", lw=1)
        ax.axhline(values.mean(), color="#2f526b", lw=1.1, ls="--")
        ax.text(.02, .97, f"平均差 {values.mean():+.2f} s／源；{sum(values < 0)}／{len(values)} 场更快",
                transform=ax.transAxes, va="top", fontsize=10)
        for rank in sorted({0, len(order)-1}):
            j = order[rank]
            ax.annotate(str(paired[j][0]), (rank, values[j]), xytext=(0, 9 if values[j] >= 0 else -13),
                        textcoords="offset points", ha="center", fontsize=8)
        ax.margins(y=.23)
    ax.set_title("自由补点 − 旧联合：同场景配对差", loc="left", pad=14)
    ax.set_xlabel("共同场景按差值排序；负数表示自由补点更快")
    ax.set_ylabel("时间差（s／源）")
    ax.grid(axis="y", alpha=.18)
    ax = axes[1, 1]
    counts = sorted({r["source_count"] for r in rows})
    for n in names:
        complete = [r for r in rows if r["strategy"] == n and r.get("completed")]
        x, y = [], []
        for count in counts:
            group = [r["time_s"] for r in complete if r["source_count"] == count]
            if group:
                x.append(count)
                y.append(np.mean(group))
        ax.plot(x, y, "o-", lw=1.8, color=PALETTE.get(n), label=SHORT.get(n, n))
    ax.set_xticks(counts)
    ax.set_title("同源数组的完整总时间", loc="left", pad=14)
    ax.set_xlabel("实际源数 N；分组关联不等于单因素因果效应")
    ax.set_ylabel("完整总时间（s／场）")
    ax.grid(axis="y", alpha=.18)
    ax.legend(frameon=False, fontsize=9)
    source_count = sum(r["source_count"] for r in rows if r["strategy"] == names[0])
    title = "绘图测试：开发样本，不能作为新验证成绩" if smoke else f"第四问自由补点 · {len(data.get('seeds', []))} 个共同场景／每策略 {source_count} 个源"
    fig.suptitle(title, fontsize=19, ha="left", x=.055, y=.985)
    fig.text(.055, .018, "本地合成；包含定位、接近、成功／失败清除。散点只包含完成任务，失败场数另列；若有失败，不给出完整组平均成绩。", fontsize=9, color="#60788c")
    fig.tight_layout(rect=[.025, .045, .99, .95], h_pad=3.5, w_pad=3)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def paired_rows(rows):
    a = {r["seed"]: r for r in rows if r["strategy"] == "free_joint" and r.get("completed")}
    b = {r["seed"]: r for r in rows if r["strategy"] == "old_joint" and r.get("completed")}
    return [(seed, a[seed], b[seed]) for seed in sorted(a.keys() & b.keys())]


def select_cases(data):
    selected = {}
    def tag(seed, title):
        selected.setdefault(int(seed), []).append(title)
    pairs = paired_rows(data["rows"])
    if pairs:
        delta = lambda row: row[1]["per_source_s"]-row[2]["per_source_s"]
        worst = max(pairs, key=delta)
        best = min(pairs, key=delta)
        tag(worst[0], "相对旧联合退步最多" if delta(worst) > 0 else "相对旧联合改善最少（本组无退步）")
        tag(best[0], "相对旧联合改善最多" if delta(best) < 0 else "相对旧联合退步最少（本组无改善）")
        tag(max(pairs, key=lambda p: p[1]["per_source_s"])[0], "自由补点最慢")
    return selected


def load_history(row, directory):
    preferred = directory / "replay_histories" / row["strategy"] / f"{row['seed']}.json.gz"
    candidates = [preferred]
    if row.get("history_file"):
        candidates.append(Path(row["history_file"]))
    candidates += [directory / "validate_histories" / row["strategy"] / f"{row['seed']}.json.gz"]
    for path in candidates:
        if not path.exists():
            continue
        result = read(path)
        summary = result["summary"]
        if abs(summary["time_s"]-row["time_s"]) > 1e-6 or summary["command_count"] != row["command_count"]:
            continue
        result["read_from"] = str(path.resolve())
        return result
    raise FileNotFoundError(f"No matching actual history for {row['strategy']} / {row['seed']}")


def draw_map(ax, history, extent):
    stops = physical_stops(history)
    rows = commands(history)
    ax.add_patch(Circle((0, 0), 1800, fill=False, ls="--", lw=1.1, ec="#b4c4d0", zorder=0))
    segments = [[s["arrival"]["from"], s["position"]] for s in stops if s["move_m"] > 1]
    colors = [KINDS[s["kind"]][1] for s in stops if s["move_m"] > 1]
    if segments:
        ax.add_collection(LineCollection(segments, colors=colors, linewidths=1.25, alpha=.60, zorder=2))
    for kind, (_, color, marker) in KINDS.items():
        selected = [s for s in stops[1:] if s["kind"] == kind]
        if selected:
            p = np.array([s["position"] for s in selected])
            ax.scatter(p[:, 0], p[:, 1], s=20, c=color, marker=marker, alpha=.85, zorder=4,
                       edgecolors="white", linewidths=.45)
    shared = [s for s in stops[1:] if s["shared"]]
    if shared:
        p = np.array([s["position"] for s in shared])
        ax.scatter(p[:, 0], p[:, 1], s=75, facecolors="none", edgecolors="#9365aa", linewidths=1.2, zorder=5)
    good = [s for s in stops if s["successful_clear"]]
    bad = [s for s in stops if s["failed_clear"]]
    for group, color, marker, size in ((good, "#1d7862", "*", 64), (bad, "#c45767", "x", 35)):
        if group:
            p = np.array([s["position"] for s in group])
            ax.scatter(p[:, 0], p[:, 1], s=size, c=color, marker=marker, zorder=6, linewidths=1.0)
    # Use a small consistent orientation glyph; drawing every 1–1.5 km radio
    # half-disk would obscure the motion history and can look like belief truth.
    source_labels = []
    for source in history["truth"]:
        q = np.array([source["x"], source["y"]])
        ax.scatter(*q, s=14, marker="D", fc="#364f62", ec="white", linewidths=.4, zorder=7)
        if source["directional"]:
            angle = float(source["orientation"])
            ax.add_patch(Wedge(q, 90, np.degrees(angle)-90, np.degrees(angle)+90,
                               facecolor="#5b7284", alpha=.12, edgecolor="none", zorder=1))
            end = q+140*np.array([np.cos(angle), np.sin(angle)])
            ax.annotate("", end, q, arrowprops=dict(arrowstyle="->", color="#526a7c", lw=1.0), zorder=7)
        source_labels.append(ax.annotate(f"C{source['channel']}", q, xytext=(6, 7), textcoords="offset points", fontsize=8,
                    color="#3c5569", zorder=8, bbox=dict(fc="white", ec="none", alpha=.68, pad=.3),
                    arrowprops=dict(arrowstyle="-", color="#93a6b4", lw=.55, shrinkA=1, shrinkB=3)))
    # At most fourteen arrival labels per map, with minimum spatial separation.
    numbered = []
    stride = max(1, int(np.ceil(max(1, len(stops)-1)/14)))
    indices = list(range(1, len(stops), stride))
    if stops and len(stops)-1 not in indices:
        indices.append(len(stops)-1)
    for i in indices:
        stop = stops[i]
        q = stop["position"]
        if any(np.linalg.norm(q-p) < extent*.065 for p in numbered):
            continue
        numbered.append(q)
        offset = (5, -11) if i % 2 else (-26, 7)
        ax.annotate(f"a{stop['arrival']['index']+1}", q, xytext=offset, textcoords="offset points",
                    fontsize=7.5, color=KINDS[stop["kind"]][1], zorder=9,
                    bbox=dict(fc="white", ec="none", alpha=.70, pad=.2))
    moving = [s for s in stops if s["move_m"] > 200]
    for stop in moving[::max(1, len(moving)//12)]:
        a = np.asarray(stop["arrival"]["from"])
        d = stop["position"]-a
        ax.annotate("", a+.62*d, a+.50*d,
                    arrowprops=dict(arrowstyle="->", color=KINDS[stop["kind"]][1], lw=.8, alpha=.75), zorder=3)
    ax.scatter(0, 0, s=44, marker="o", fc="white", ec="#243f54", zorder=9)
    ax.annotate("起点", (0, 0), xytext=(6, -12), textcoords="offset points", fontsize=8)
    if rows:
        ax.scatter(*rows[-1]["position"], s=45, marker="s", fc="#183c51", ec="white", lw=.6, zorder=9)
    ax.set_xlim(-extent, extent)
    ax.set_ylim(-extent, extent)
    ax.set_aspect("equal")
    ax.set_xlabel("x（m）")
    ax.set_ylabel("y（m）")
    ax.grid(alpha=.12)
    # Nearby emitters can make fixed C labels overlap. Move only the labels;
    # leader lines and source glyph coordinates remain tied to recorded truth.
    ax.figure.canvas.draw()
    renderer = ax.figure.canvas.get_renderer()
    occupied = [t.get_window_extent(renderer).expanded(1.08, 1.15)
                for t in ax.texts if t not in source_labels and t.get_text()]
    for label in source_labels:
        best, smallest = (6, 7), float("inf")
        for offset in ((6, 7), (6, -13), (-24, 7), (-24, -13),
                       (10, 19), (10, -23), (-30, 19), (-30, -23)):
            label.set_position(offset)
            box = label.get_window_extent(renderer).expanded(1.10, 1.20)
            overlaps = sum(box.overlaps(other) for other in occupied)
            if overlaps < smallest:
                best, smallest = offset, overlaps
            if overlaps == 0:
                break
        label.set_position(best)
        occupied.append(label.get_window_extent(renderer).expanded(1.10, 1.20))


def draw_progress(ax, history, end_s):
    times, discovered, cleared = [0.0], [0], [0]
    seen, done = set(), set()
    for row in commands(history):
        if answer(row) in ("direction", "near"):
            seen.add(row["channel"])
        if row["kind"] == "clear" and answer(row) == "success":
            done.add(row["channel"])
        times.append(row["time_s"])
        discovered.append(len(seen))
        cleared.append(len(done))
    ax.step(times, discovered, where="post", color="#4d86af", lw=1.8, label="已发现")
    ax.step(times, cleared, where="post", color="#389078", lw=1.8, label="已清除")
    if done:
        last_clear = max(r["time_s"] for r in commands(history) if r["kind"] == "clear" and answer(r) == "success")
        if times[-1]-last_clear > 1:
            ax.axvspan(last_clear, times[-1], color="#e7bf73", alpha=.20)
            ax.annotate(f"无源确认尾段 {times[-1]-last_clear:.0f}s", ((last_clear+times[-1])/2, .4),
                        ha="center", fontsize=8, color="#9b7633")
    ax.set_xlim(0, end_s*1.02)
    ax.set_ylim(-.2, len(history["truth"])+1)
    ax.set_ylabel("源数")
    ax.set_xlabel("累计任务时间（s），两图使用相同时间尺度")
    ax.set_yticks(sorted({0, len(history["truth"])}))
    ax.grid(axis="y", alpha=.15)
    ax.legend(loc="upper left", fontsize=8, frameon=False, ncol=2)


def case_plot(seed, histories, tags, output, smoke=False):
    names = [n for n in ("free_joint", "old_joint") if n in histories]
    if len(names) < 2:
        raise ValueError("Paired history requires new dynamic_joint and frozen old_joint")
    truth = histories[names[0]]["truth"]
    if histories[names[1]]["truth"] != truth:
        raise ValueError("Paired plots must use exactly the same scene")
    extent = max(2040., max(abs(v) for n in names for row in commands(histories[n]) for v in row["position"])+150)
    end_s = max(histories[n]["summary"]["time_s"] for n in names)
    fig = plt.figure(figsize=(16.5, 12.5))
    grid = fig.add_gridspec(2, 2, height_ratios=[4.0, 1.0], left=.07, right=.97,
                           bottom=.14, top=.86, hspace=.48, wspace=.17)
    metrics = {}
    for i, name in enumerate(names):
        history = histories[name]
        summary = history["summary"]
        m = motion_metrics(history)
        metrics[name] = m
        ax = fig.add_subplot(grid[0, i])
        draw_map(ax, history, extent)
        short = SHORT.get(name, name)
        score = summary.get("per_source_s")
        score_text = f"{score:.2f} s／源" if score is not None else "未完成，不能计完整成绩"
        ax.set_title(f"{short}｜{score_text}｜总计 {summary['time_s']:.0f} s", fontsize=13, loc="left", pad=12)
        caption = (f"实际移动 {m['movement_m']/1000:.2f} km；停点 {m['physical_stops']} 个；共享 {m['shared_stops']} 个；中途 {m['intermediate_stops']} 个\n"
                   f"已知源失联 {m['known_no_signal']} 次；清除失败 {m['clear_failures']} 次；实际清除 {m['clear_successes']}／{len(truth)}")
        ax.text(0., -.13, caption, transform=ax.transAxes, fontsize=9, va="top", color="#557085", linespacing=1.6)
        draw_progress(fig.add_subplot(grid[1, i]), history, end_s)
    delta = histories["free_joint"]["summary"]["per_source_s"]-histories["old_joint"]["summary"]["per_source_s"]
    title = f"完整运动历史 · 种子 {seed} · {len(truth)} 个源 · 新−旧 {delta:+.2f} s／源"
    fig.suptitle(title, fontsize=19, x=.055, ha="left", y=.965)
    prefix = "绘图测试，开发数据，非验证成绩｜" if smoke else ""
    fig.text(.055, .923, prefix+"；".join(tags), fontsize=10, color="#5b7283")
    handles = [Line2D([0], [0], marker=marker, color=color, lw=1.3, markersize=5, label=label)
               for label, color, marker in KINDS.values()]
    handles += [Line2D([0], [0], marker="o", color="#9365aa", markerfacecolor="none", ls="", markersize=8, label="实际共享停点"),
                Line2D([0], [0], marker="*", color="#1d7862", ls="", markersize=9, label="清除成功"),
                Line2D([0], [0], marker="x", color="#c45767", ls="", label="清除失败"),
                Line2D([0], [0], marker="s", color="#183c51", ls="", label="终点")]
    fig.legend(handles=handles, ncol=7, loc="lower center", bbox_to_anchor=(.5, .065), fontsize=9, frameon=False)
    fig.text(.055, .039, "共享按真实同点动作定义；中途候选只有实际停下测量才画出。a编号为实际动作序号，仅抽取部分以避免重叠。", fontsize=9, color="#617b8e")
    fig.text(.055, .020, "灰色C编号和短箭头是评估真值（位置／发射朝向），策略不可见；虚线为源域R=1800m。图中未把行进线段计作扫描。", fontsize=9, color="#617b8e")
    path = output / f"case_{seed}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return dict(seed=seed, tags=tags, image=str(path.resolve()), metrics=metrics,
                history_files={name: histories[name].get("read_from") for name in names})


def unknown_overlay(seed, history, output):
    """Last recorded nonempty unknown map; it is not a coverage certificate."""
    selected = None
    for frame in reversed(history.get("frames") or []):
        maps = [d for d in frame.get("unknown_domains", []) if d.get("cells") and not d.get("certified")]
        if maps:
            selected = frame, max(maps, key=lambda d: d.get("remaining_mass", 0.0))
            break
    if selected is None:
        return None
    frame, domain = selected
    cells = np.asarray(domain["cells"])
    fig, ax = plt.subplots(figsize=(9.5, 8.5))
    ax.add_patch(Circle((0, 0), 1800, fill=False, ls="--", ec="#a7bdcc"))
    scatter = ax.scatter(cells[:, 0], cells[:, 1], c=cells[:, 2], s=120, marker="s",
                         cmap="YlOrRd", vmin=0, vmax=1, alpha=.82, linewidths=0)
    prefix = [r for r in commands(history) if r["index"] <= frame["index"]]
    found = {r["channel"] for r in prefix if answer(r) in ("direction", "near")}
    positions = np.asarray([[0, 0]]+[r["position"] for r in prefix])
    ax.plot(positions[:, 0], positions[:, 1], color="#647e91", lw=.7, alpha=.27)
    negative = np.asarray(domain.get("negative_points", [])).reshape(-1, 2)
    if len(negative):
        ax.scatter(negative[:, 0], negative[:, 1], c="#416e96", marker="x", s=24, label="该频道真实负观测")
    ax.scatter(*frame["position"], marker="s", s=55, c="#203f57", label="当前停点")
    extent = max(2050., np.max(np.abs(positions))+100)
    ax.set_xlim(-extent, extent)
    ax.set_ylim(-extent, extent)
    ax.set_aspect("equal")
    ax.set_xlabel("x（m）")
    ax.set_ylabel("y（m）")
    ax.set_title(f"最后有效未知方向图 · 种子 {seed} · 频道 {domain['channel']}\n动作 a{frame['index']+1}，t={frame['time_s']:.0f}s，剩余离散质量 {domain['remaining_mass']:.3%}", loc="left", fontsize=13, pad=13)
    ax.grid(alpha=.12)
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    fig.colorbar(scatter, ax=ax, shrink=.72, pad=.025, label="该位置剩余朝向比例（非源存在概率）")
    fig.text(.075, .036, "每格为该位置仍相容的离散朝向比例；粗网格只用于规划，清空也不能证明全域无源。", fontsize=9, color="#5d778a")
    if len(found) >= 16:
        fig.text(.075, .016, "此时已发现16个源，依据题目源数上限可结束搜索；几何网格因此可以保留非零余量。", fontsize=9, color="#5d778a")
    fig.tight_layout(rect=[.01, .055, .99, .99])
    path = output / f"unknown_{seed}_c{domain['channel']}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return dict(image=str(path.resolve()), channel=domain["channel"], action=frame["index"]+1,
                time_s=frame["time_s"], remaining_mass=domain["remaining_mass"],
                discovered_source_count=len(found), source_count_upper_bound_reached=len(found) >= 16)


def smoke_data():
    files = [BASE/"outputs"/"current_smoke"/"free_joint"/"20285000.json.gz",
             BASE/"outputs"/"harness_check"/"old_joint"/"20285000.json.gz"]
    histories = [read(path) for path in files]
    rows, summary = [], []
    for path, history in zip(files, histories):
        row = dict(history["summary"], history_file=str(path.resolve()))
        rows.append(row)
        summary.append(dict(strategy=row["strategy"], costs_per_source=row["costs_per_source"],
                            failed_count=int(not row["completed"])))
    return dict(stage="smoke_visualization_only", rows=rows, summary=summary, seeds=[20285000])


def certificate_plot(history, output):
    """Display a real continuously certified negative-observation support."""
    from scipy.spatial import Delaunay, ConvexHull
    from question4.zigzag_study.study import certificate_details, triangle_min_radius
    channels = history.get('coverage', {}).get('channels', {})
    candidates = [c for c in channels.values() if c.get('certified') and c.get('proof_points')]
    if not candidates:
        return None
    c = min(candidates, key=lambda v: len(v['proof_points']))
    points = np.array(c['proof_points'])
    proof = certificate_details(points)
    if not proof['certified']:
        raise AssertionError('Saved support failed the independent continuous certificate')
    triangles = points[Delaunay(points).simplices]
    triangles = triangles[triangle_min_radius(triangles) <= 1800.+1e-7]
    edge = np.linalg.norm(triangles-np.roll(triangles, -1, axis=1), axis=2).max()
    hull = ConvexHull(points)
    inradius = np.min(-hull.equations[:, 2])
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    ax = axes[0]
    for tri in triangles:
        loop = np.vstack((tri, tri[0]))
        ax.fill(tri[:, 0], tri[:, 1], color='#529b86', alpha=.10)
        ax.plot(loop[:, 0], loop[:, 1], color='#4d9985', alpha=.55, lw=.8)
    ax.scatter(points[:, 0], points[:, 1], s=20, color='#24516e', zorder=4)
    ax.add_patch(Circle((0, 0), 1800, fill=False, ec='#b06b58', lw=2))
    ax.set_title(f"频道{c['channel']}实际负反馈构成的连续覆盖证书\n{len(points)}个支持点；相关三角形最长边 {edge:.1f}m", loc='left')
    ax.set_aspect('equal'); ax.grid(alpha=.15)
    ax.set_xlabel('x（m）'); ax.set_ylabel('y（m）')
    ax = axes[1]
    # A representative continuous triangle illustrates why all positions and
    # all emitting orientations are covered, independently of map resolution.
    tri = max(triangles, key=lambda t: abs((t[1, 0]-t[0, 0])*(t[2, 1]-t[0, 1])
                                          -(t[1, 1]-t[0, 1])*(t[2, 0]-t[0, 0])))
    g = tri.mean(axis=0)
    loop = np.vstack((tri, tri[0]))
    ax.fill(tri[:, 0], tri[:, 1], color='#529b86', alpha=.12)
    ax.plot(loop[:, 0], loop[:, 1], color='#4d9985', lw=1.5)
    ax.scatter(tri[:, 0], tri[:, 1], s=50, color='#24516e')
    for j, q in enumerate(tri):
        ax.plot([g[0], q[0]], [g[1], q[1]], ':', color='#849cad')
        ax.annotate(f'q{j+1}', q, xytext=(7, 7), textcoords='offset points')
    ax.scatter(*g, s=55, color='#bb7657')
    ax.annotate('任意待检位置 g', g, xytext=(7, -18), textcoords='offset points')
    extent = max(np.ptp(tri, axis=0))*1.25
    ax.set_xlim(g[0]-extent/2, g[0]+extent/2)
    ax.set_ylim(g[1]-extent/2, g[1]+extent/2)
    ax.set_aspect('equal'); ax.grid(alpha=.15)
    ax.set_title('三角形内部的每个位置，三顶点均在1000m内\n任意180°发射半平面至少包含其中一个顶点', loc='left')
    fig.suptitle(f"种子{history['summary']['seed']}：结束补盲的依据是连续几何证书", fontsize=16, x=.07, ha='left')
    fig.text(.07, .025, f'支持点凸包内切半径 {inradius:.2f}m ≥ 1800m；图中三角形为真实负反馈支持点生成，未把计划点当作实测证据。', fontsize=10)
    fig.tight_layout(rect=[.02, .055, .99, .93])
    fig.savefig(output, dpi=150); plt.close(fig)
    return dict(channel=c['channel'], support_count=len(points), max_edge_m=float(edge), hull_inradius_m=float(inradius))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=BASE/"outputs"/"validate.json")
    parser.add_argument("--output", type=Path, default=BASE/"outputs"/"figures")
    parser.add_argument("--seeds", type=int, nargs="*")
    parser.add_argument("--unknown-overlay", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Render existing development sample; not a validation result")
    args = parser.parse_args()
    configure()
    output = args.output / "smoke" if args.smoke else args.output
    output.mkdir(parents=True, exist_ok=True)
    data = smoke_data() if args.smoke else read(args.input)
    selected = ({seed: ["指定复核样本"] for seed in args.seeds} if args.seeds else select_cases(data))
    overview(data, output/"overview.png", smoke=args.smoke)
    manifest = dict(scope="development rendering test" if args.smoke else data.get("scope", "local synthetic complete mission"),
                    source_file=None if args.smoke else str(args.input.resolve()),
                    overview=str((output/"overview.png").resolve()), cases=[], definitions={
                        "metric": "每场完整总时间／实际源数，再对共同场景等权平均",
                        "truth": "源坐标和朝向只用于离线评估显示，不重建或更改动作",
                        "motion": "同位置连续扫描合并为停点；移动只按实际命令from→position",
                        "shared": "实际同点完成搜索和服务，或服务至少两个已知频道"})
    for seed, tags in selected.items():
        histories = {r["strategy"]: load_history(r, args.input.parent) for r in data["rows"]
                     if r["seed"] == seed and r["strategy"] in ("free_joint", "old_joint")}
        if len(histories) < 2:
            continue
        case = case_plot(seed, histories, tags, output, smoke=args.smoke)
        if args.unknown_overlay:
            case["unknown_overlay"] = unknown_overlay(seed, histories["free_joint"], output)
        manifest["cases"].append(case)
    write(output/"diagnostics_manifest.json", manifest)
    print(json.dumps(dict(output=str(output.resolve()), cases=len(manifest["cases"])), ensure_ascii=False))


if __name__ == "__main__":
    main()

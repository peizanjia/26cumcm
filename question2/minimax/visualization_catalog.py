"""Build a broad, paper-ready visualization catalog for question 2."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Wedge
import numpy as np

from .mechanistic import (
    ANGLE_ERROR, ARENA_RADIUS, RECEIVE_MAX, RECEIVE_MIN,
    clip_disk, clip_wedge, initial_region, polygon_area,
    polygon_min_enclosing_circle, possible_bearings,
    reception_margin_from_supports, reception_supports, robust_score, wrap_angle,
)


plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Noto Sans CJK SC", "Microsoft YaHei", "SimHei", "Arial Unicode MS", "Hiragino Sans GB", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
})

POLICIES = ["minimax_mec", "analytic_far_range", "fixed_750_375"]
LABELS = {
    "minimax_mec": "Minimax MEC",
    "analytic_far_range": "远场解析",
    "fixed_750_375": "固定点",
}
COLORS = {
    "minimax_mec": "#3b6ea8",
    "analytic_far_range": "#e28e2c",
    "fixed_750_375": "#59a14f",
}
MARKERS = {"minimax_mec": "o", "analytic_far_range": "s", "fixed_750_375": "^"}


def load_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _poly(axis: plt.Axes, points: np.ndarray, color: str, alpha: float,
          label: str | None = None, zorder: int = 2) -> None:
    axis.fill(points[:, 0], points[:, 1], color=color, alpha=alpha,
              label=label, zorder=zorder)
    axis.plot(points[:, 0], points[:, 1], color=color, lw=1.2, zorder=zorder + 0.1)


def problem_geometry(summary: dict, output: Path) -> None:
    station = np.zeros(2)
    bearing = 0.0
    q = np.asarray(summary["representative"]["center"]["point_m"], float)
    first = initial_region(station, bearing, 128)
    score = robust_score(first, q, 1.0, 128)
    posterior = clip_wedge(clip_disk(first, q, RECEIVE_MAX, 128),
                           q, score.worst_bearing_rad)
    mec_center, mec_radius = polygon_min_enclosing_circle(posterior)
    nominal = np.array([1000.0, 0.0])

    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.9), constrained_layout=True)
    for axis in axes:
        axis.set_aspect("equal")
        axis.grid(alpha=0.14)
        axis.set_xlabel("x / m")
    axes[0].add_patch(Wedge(station, 1500, -1, 1, color="#9ecae1", alpha=.28))
    _poly(axes[0], first, "#3b6ea8", .28, "第一次定位域")
    axes[0].plot(*station, "ko", label="第一监测点")
    axes[0].plot(*nominal, marker="X", color="#d1495b", ms=8, label="示意真值")
    axes[0].annotate("测向 ±1°", xy=(1100, 30), xytext=(500, 420),
                     arrowprops=dict(arrowstyle="->", color="0.35"))
    axes[0].set_title("(a) 第一次测向后的集合")
    axes[0].legend(frameon=False, loc="upper left")

    supports = reception_supports(station, bearing, .05)
    grid = np.linspace(-1050, 1050, 151)
    xx, yy = np.meshgrid(grid, grid)
    pts = np.column_stack((xx.ravel(), yy.ravel()))
    margin = np.empty(len(pts))
    for start in range(0, len(pts), 3000):
        part = pts[start:start + 3000]
        dist = np.linalg.norm(part[:, None, :] - supports.points[None, :, :], axis=2)
        margin[start:start + len(part)] = np.min(supports.radii - dist, axis=1)
    margin = margin.reshape(xx.shape)
    axes[1].contourf(xx, yy, margin, levels=[0, max(1, margin.max())],
                     colors=["#cce8df"], alpha=.8)
    axes[1].contour(xx, yy, margin, levels=[0], colors=["#238b73"], linewidths=1.4)
    axes[1].plot(*station, "ko", label="第一监测点")
    axes[1].plot(*q, marker="*", color="#e15759", ms=13, label="Minimax 第二点")
    axes[1].set_title("(b) 完整可靠接收域与选点")
    axes[1].legend(frameon=False, loc="upper left")

    _poly(axes[2], first, "#bab0ac", .22, "第一次定位域")
    _poly(axes[2], posterior, "#4e79a7", .55, "最坏后验区域")
    axes[2].add_patch(Circle(mec_center, mec_radius, fill=False, color="#e15759",
                             lw=2, label="最小覆盖圆"))
    axes[2].plot(*station, "ko")
    axes[2].plot(*q, marker="*", color="#f28e2b", ms=13, label="第二监测点")
    axes[2].set_title(f"(c) 最坏后验与 MEC：R={mec_radius:.2f} m")
    axes[2].legend(frameon=False, loc="upper left")
    axes[0].set_xlim(-150, 1650); axes[0].set_ylim(-800, 800)
    axes[1].set_xlim(-1050, 1050); axes[1].set_ylim(-1050, 1050)
    axes[2].set_xlim(-150, 1550); axes[2].set_ylim(-800, 800)
    axes[0].set_ylabel("y / m")
    save(fig, output)


def safe_mechanism(output: Path) -> None:
    station = np.array([1700.0, 0.0])
    bearing = 0.0
    supports = reception_supports(station, bearing, .05)
    grid_x = np.linspace(650, 2750, 181)
    grid_y = np.linspace(-1050, 1050, 181)
    xx, yy = np.meshgrid(grid_x, grid_y)
    pts = np.column_stack((xx.ravel(), yy.ravel()))
    margin = np.empty(len(pts))
    for start in range(0, len(pts), 2500):
        part = pts[start:start + 2500]
        distance = np.linalg.norm(part[:, None, :] - supports.points[None, :, :], axis=2)
        margin[start:start + len(part)] = np.min(supports.radii - distance, axis=1)
    margin = margin.reshape(xx.shape)
    first = initial_region(station, bearing, 128)

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.2), constrained_layout=True)
    axis = axes[0]
    _poly(axis, first, "#4e79a7", .42, "边界裁切后的目标集合")
    chosen = [0, len(supports.points)//2, len(supports.points)-1]
    circle_colors = ["#e15759", "#59a14f", "#f28e2b"]
    for sequence, (idx, color) in enumerate(zip(chosen, circle_colors)):
        x, rho = supports.points[idx], supports.radii[idx]
        axis.add_patch(Circle(x, rho, fill=False, color=color, lw=1.2,
                              label=("代表性约束圆 B(x,ρmin)"
                                     if sequence == 0 else None)))
        axis.plot(*x, "o", color=color, ms=5)
    axis.plot(*station, "ko", label="第一监测点")
    axis.add_patch(Circle((0, 0), ARENA_RADIUS, fill=False, ls="--", color="0.5"))
    axis.set_xlim(650, 2750); axis.set_ylim(-1050, 1050); axis.set_aspect("equal")
    axis.set_title("(a) 每个可能目标位置给出一个接收约束圆")
    axis.set_xlabel("x / m"); axis.set_ylabel("y / m")
    axis.grid(alpha=.15); axis.legend(frameon=False, fontsize=8, loc="lower left")

    axis = axes[1]
    axis.contourf(xx, yy, margin, levels=[0, max(1, margin.max())],
                  colors=["#cce8df"], alpha=.9)
    axis.contour(xx, yy, margin, levels=[0], colors=["#238b73"], linewidths=1.7)
    axis.contour(xx, yy, margin, levels=[100, 250, 500],
                 colors=["#6aa89a"], linewidths=.8, alpha=.65)
    axis.plot(*station, "ko")
    axis.text(station[0]+25, station[1]+30, "S₁")
    axis.annotate("所有约束圆的交集\n即完整可靠接收域", xy=(1500, 650),
                  xytext=(800, 900), arrowprops=dict(arrowstyle="->", color="0.3"))
    axis.add_patch(Circle((0, 0), ARENA_RADIUS, fill=False, ls="--", color="0.5"))
    axis.set_xlim(650, 2750); axis.set_ylim(-1050, 1050); axis.set_aspect("equal")
    axis.set_title("(b) 交集保留边界向外新增候选区域")
    axis.set_xlabel("x / m"); axis.grid(alpha=.15)
    save(fig, output)


def mec_theory(output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.2), constrained_layout=True)
    cases = []
    obtuse = np.array([[-1.8, 0], [1.8, 0], [.2, 1.0]])
    cases.append((obtuse, "(a) 两点支撑：最长边为直径"))
    acute = np.array([[0, 1.7], [-1.5, -1], [1.5, -1]])
    cases.append((acute, "(b) 三点支撑：三点外接圆"))
    counter = np.array([[1, 0], [-.5, math.sqrt(3)/2], [-.99, 0],
                        [-.5, -math.sqrt(3)/2]])
    cases.append((counter, "(c) 固定直径端点法的反例"))
    for index, (points, title) in enumerate(cases):
        axis = axes[index]
        center, radius = polygon_min_enclosing_circle(points)
        axis.fill(points[:, 0], points[:, 1], color="#9ecae1", alpha=.35)
        axis.plot(np.r_[points[:, 0], points[0, 0]],
                  np.r_[points[:, 1], points[0, 1]], color="#3b6ea8")
        axis.scatter(points[:, 0], points[:, 1], color="#3b6ea8", s=35, zorder=3)
        axis.add_patch(Circle(center, radius, fill=False, color="#e15759", lw=2,
                              label="正确 MEC"))
        axis.plot(*center, "+", color="#e15759", ms=9, mew=1.5)
        if index == 2:
            d = np.linalg.norm(points[:, None] - points[None, :], axis=2)
            i, j = np.unravel_index(np.argmax(d), d.shape)
            wrong_center = (points[i] + points[j]) / 2
            wrong_radius = d[i, j] / 2
            axis.add_patch(Circle(wrong_center, wrong_radius, fill=False,
                                  color="#f28e2b", ls="--", lw=1.7,
                                  label="固定直径圆（漏点）"))
            axis.plot(points[[i, j], 0], points[[i, j], 1], color="#f28e2b",
                      lw=2, label="全局直径")
        axis.set_title(title)
        axis.set_aspect("equal"); axis.grid(alpha=.15)
        axis.set_xlim(min(points[:, 0].min(), center[0]-radius)-.25,
                      max(points[:, 0].max(), center[0]+radius)+.25)
        axis.set_ylim(min(points[:, 1].min(), center[1]-radius)-.25,
                      max(points[:, 1].max(), center[1]+radius)+.25)
        axis.legend(frameon=False, fontsize=8, loc="upper right")
    save(fig, output)


def minimax_flowchart(output: Path) -> None:
    fig, axis = plt.subplots(figsize=(9.2, 11.0), constrained_layout=True)
    axis.set_xlim(0, 10); axis.set_ylim(0, 14); axis.axis("off")
    nodes = [
        (5, 13.2, 5.6, .8, "输入：S₁、第一次示向度 θ̂₁、误差 ±1°", "#dceaf7"),
        (5, 11.8, 5.6, .8, "构造第一次可行域  X₁=D∩B(S₁,1500)∩W₁", "#dceaf7"),
        (5, 10.4, 6.6, .95, "按真实径向区间计算可靠接收域\nCrec=⋂ B(x,max{1000,‖x−S₁‖})", "#dff1e8"),
        (5, 8.9, 5.6, .8, "生成全角度候选网格并筛选 q∈Crec", "#dff1e8"),
        (5, 7.45, 5.8, .9, "对每个 q 枚举可能第二示向度 z", "#fae5c8"),
        (5, 6.0, 6.1, .9, "裁剪后验区域  P₂(q,z)=X₁∩B(q,1500)∩W₂", "#fae5c8"),
        (5, 4.55, 5.8, .9, "枚举两点/三点支撑圆，求 R_MEC(P₂)", "#f6d6d8"),
        (5, 3.1, 5.8, .9, "取最坏示向度  J_R(q)=sup_z R_MEC(P₂)", "#f6d6d8"),
        (5, 1.65, 6.2, .9, "最小化 J_R；面积与移动距离用于同分决策", "#e7dcf2"),
        (5, .4, 7.2, .8, "输出：q*、5%近优候选集 C₅%、阈值候选集 C₂₀", "#e7dcf2"),
    ]
    for x, y, w, h, text, color in nodes:
        patch = FancyBboxPatch((x-w/2, y-h/2), w, h,
                               boxstyle="round,pad=.025,rounding_size=.12",
                               facecolor=color, edgecolor="#44546a", lw=1.1)
        axis.add_patch(patch)
        axis.text(x, y, text, ha="center", va="center", fontsize=11)
    for upper, lower in zip(nodes[:-1], nodes[1:]):
        axis.add_patch(FancyArrowPatch((5, upper[1]-upper[3]/2),
                                       (5, lower[1]+lower[3]/2),
                                       arrowstyle="-|>", mutation_scale=13,
                                       color="#44546a", lw=1.15))
    axis.text(8.65, 7.0, "对全部候选点循环", rotation=90,
              ha="center", va="center", color="#6b4c7a")
    axis.add_patch(FancyArrowPatch((7.9, 3.1), (7.9, 8.9),
                                   connectionstyle="arc3,rad=-.08",
                                   arrowstyle="-|>", mutation_scale=13,
                                   color="#6b4c7a", lw=1.1))
    axis.set_title("边界感知 Minimax-MEC 第二监测点算法流程", fontsize=16, pad=12)
    save(fig, output)


def sample_scenarios(rows: list[dict], output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.3), constrained_layout=True)
    for axis, scenario, title in zip(
        axes, ("global_random", "edge_outward_stress"),
        ("(a) 全局随机样本", "(b) 边界向外压力样本")):
        part = [r for r in rows if r["scenario"] == scenario
                and r["policy"] == "minimax_mec"]
        for row in part:
            sx, sy = float(row["station_x_m"]), float(row["station_y_m"])
            tx, ty = float(row["target_x_m"]), float(row["target_y_m"])
            axis.plot([sx, tx], [sy, ty], color="0.75", lw=.55, alpha=.7)
        axis.scatter([float(r["station_x_m"]) for r in part],
                     [float(r["station_y_m"]) for r in part], s=23,
                     color="#4e79a7", label="第一监测点")
        axis.scatter([float(r["target_x_m"]) for r in part],
                     [float(r["target_y_m"]) for r in part], s=25,
                     marker="x", color="#e15759", label="干扰源真值")
        axis.add_patch(Circle((0, 0), ARENA_RADIUS, fill=False, ls="--", color="0.45"))
        axis.set_xlim(-1900, 1900); axis.set_ylim(-1900, 1900)
        axis.set_aspect("equal"); axis.grid(alpha=.15); axis.set_title(title)
        axis.set_xlabel("x / m"); axis.legend(frameon=False)
    axes[0].set_ylabel("y / m")
    save(fig, output)


def _bearing_lookup(rows: list[dict]) -> dict[tuple[str, int], float]:
    result = {}
    for row in rows:
        if row["policy"] != "fixed_750_375":
            continue
        dx = float(row["q_x_m"]) - float(row["station_x_m"])
        dy = float(row["q_y_m"]) - float(row["station_y_m"])
        result[(row["scenario"], int(row["trial"]))] = math.atan2(dy, dx) - math.atan2(375, 750)
    return result


def strategy_actions(rows: list[dict], output: Path) -> None:
    bearings = _bearing_lookup(rows)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.3), constrained_layout=True)
    for axis, scenario, title in zip(
        axes, ("global_random", "edge_outward_stress"),
        ("(a) 全局随机样本", "(b) 边界向外压力样本")):
        for policy in POLICIES:
            part = [r for r in rows if r["scenario"] == scenario and r["policy"] == policy]
            local = []
            for row in part:
                b = bearings[(scenario, int(row["trial"]))]
                d = np.array([float(row["q_x_m"])-float(row["station_x_m"]),
                              float(row["q_y_m"])-float(row["station_y_m"])])
                local.append([math.cos(b)*d[0]+math.sin(b)*d[1],
                              -math.sin(b)*d[0]+math.cos(b)*d[1]])
            local = np.asarray(local)
            axis.scatter(local[:, 0], local[:, 1], s=32, alpha=.72,
                         marker=MARKERS[policy], color=COLORS[policy], label=LABELS[policy])
        axis.add_patch(Circle((0, 0), RECEIVE_MIN, fill=False, ls="--", color="0.55"))
        axis.axhline(0, color="0.7", lw=.8); axis.axvline(0, color="0.7", lw=.8)
        axis.set_aspect("equal"); axis.grid(alpha=.15); axis.set_title(title)
        axis.set_xlabel("沿第一次示向轴前进距离 / m")
        axis.set_ylabel("横向偏移 / m")
        axis.legend(frameon=False, loc="upper left")
    save(fig, output)


def candidate_landscape(candidates: list[dict], output: Path) -> None:
    configs = {
        "center": (np.zeros(2), 0.0),
        "outward edge": (np.array([1700., 0.]), 0.0),
        "near tangent": (np.array([1700., 0.]), math.pi/2),
    }
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.8), constrained_layout=True)
    for axis, name in zip(axes, configs):
        station, bearing = configs[name]
        part = [r for r in candidates if r["scenario"] == name]
        q = np.array([[float(r["x_m"]), float(r["y_m"])] for r in part])
        d = q - station
        angle = np.degrees(wrap_angle(np.arctan2(d[:, 1], d[:, 0]) - bearing))
        radius = np.linalg.norm(d, axis=1)
        score = np.array([float(r["worst_mec_radius_m"]) for r in part])
        scatter = axis.scatter(angle, radius, c=score, cmap="viridis_r", s=24)
        selected = np.array([int(r["within_5pct"]) for r in part], bool)
        axis.scatter(angle[selected], radius[selected], facecolors="none",
                     edgecolors="#e15759", s=55, lw=1.2, label="5% 近优")
        threshold = score <= 20
        if threshold.any():
            axis.scatter(angle[threshold], radius[threshold], facecolors="none",
                         edgecolors="#3b6ea8", s=35, lw=.8, label="J≤20 m")
        axis.set_xlabel("相对第一次示向方向 / °")
        axis.set_ylabel("移动距离 / m")
        axis.set_title(name)
        axis.grid(alpha=.16); axis.legend(frameon=False, fontsize=8)
        fig.colorbar(scatter, ax=axis, shrink=.8, label="最坏 MEC 半径 / m")
    save(fig, output)


def worst_bearing_profiles(summary: dict, output: Path) -> None:
    cases = [
        ("中心", np.zeros(2), 0.0, "center"),
        ("边界向外", np.array([1700., 0.]), 0.0, "outward edge"),
        ("近切向", np.array([1700., 0.]), math.pi/2, "near tangent"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.2), constrained_layout=True)
    for axis, (title, station, bearing, key) in zip(axes, cases):
        q = np.asarray(summary["representative"][key]["point_m"], float)
        first = initial_region(station, bearing, 128)
        base = clip_disk(first, q, RECEIVE_MAX, 128)
        centroid = base.mean(axis=0)
        reference = math.atan2(centroid[1]-q[1], centroid[0]-q[0])
        angles = possible_bearings(base, q, .25)
        values = []
        relative = []
        for angle in angles:
            poly = clip_wedge(base, q, float(angle))
            if len(poly):
                values.append(polygon_min_enclosing_circle(poly)[1])
                relative.append(math.degrees(wrap_angle(float(angle)-reference)))
        order = np.argsort(relative)
        relative, values = np.asarray(relative)[order], np.asarray(values)[order]
        axis.plot(relative, values, color="#3b6ea8", lw=1.7)
        idx = int(np.argmax(values))
        axis.plot(relative[idx], values[idx], "o", color="#e15759",
                  label=f"最坏值 {values[idx]:.2f} m")
        axis.axhline(20, color="#f28e2b", ls="--", lw=1, label="20 m")
        axis.set_title(title); axis.set_xlabel("第二示向度相对中心方向 / °")
        axis.set_ylabel("后验区域 MEC 半径 / m")
        axis.grid(alpha=.18); axis.legend(frameon=False, fontsize=8)
    save(fig, output)


def observed_vs_robust(rows: list[dict], output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.2), constrained_layout=True)
    for axis, scenario, title in zip(
        axes, ("global_random", "edge_outward_stress"),
        ("(a) 全局随机", "(b) 边界压力")):
        part = [r for r in rows if r["scenario"] == scenario]
        maximum = 0.0
        for policy in POLICIES:
            selected = [r for r in part if r["policy"] == policy]
            x = np.array([float(r["mec_radius_m"]) for r in selected])
            y = np.array([float(r["conditional_worst_mec_radius_m"]) for r in selected])
            maximum = max(maximum, float(x.max()), float(y.max()))
            axis.scatter(x, y, s=31, alpha=.68, marker=MARKERS[policy],
                         color=COLORS[policy], label=LABELS[policy])
        axis.plot([0, maximum*1.05], [0, maximum*1.05], ls="--", color="0.4",
                  label="观测值=最坏上界")
        axis.set_xlim(0, maximum*1.05); axis.set_ylim(0, maximum*1.05)
        axis.set_aspect("equal"); axis.grid(alpha=.17); axis.set_title(title)
        axis.set_xlabel("一次仿真观测 MEC 半径 / m")
        axis.set_ylabel("条件最坏 MEC 半径 / m")
        axis.legend(frameon=False, fontsize=8)
    save(fig, output)


def paired_histograms(rows: list[dict], output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.6, 7.8), constrained_layout=True)
    for ri, scenario in enumerate(("global_random", "edge_outward_stress")):
        part = [r for r in rows if r["scenario"] == scenario]
        lookup = {(int(r["trial"]), r["policy"]): float(r["mec_radius_m"]) for r in part}
        trials = sorted({int(r["trial"]) for r in part})
        for ci, baseline in enumerate(("analytic_far_range", "fixed_750_375")):
            improvement = np.array([lookup[(t, baseline)]-lookup[(t, "minimax_mec")]
                                    for t in trials])
            axis = axes[ri, ci]
            axis.hist(improvement, bins=max(7, int(math.sqrt(len(improvement)))),
                      color=COLORS[baseline], alpha=.75, edgecolor="white")
            axis.axvline(0, color="0.35", ls="--", lw=1)
            axis.axvline(improvement.mean(), color="#e15759", lw=1.7,
                         label=f"均值 {improvement.mean():.2f} m")
            axis.set_title(("全局随机" if ri == 0 else "边界压力")
                           + "：" + LABELS[baseline])
            axis.set_xlabel("基准策略半径 − Minimax 半径 / m")
            axis.set_ylabel("案例数")
            axis.grid(axis="y", alpha=.17); axis.legend(frameon=False)
    save(fig, output)


def movement_tradeoff(summary: dict, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.8), constrained_layout=True)
    for axis, scenario, title in zip(
        axes, ("global_random", "edge_outward_stress"),
        ("(a) 全局随机", "(b) 边界压力")):
        part = {r["policy"]: r for r in summary["summary"] if r["scenario"] == scenario}
        for policy in POLICIES:
            row = part[policy]
            x, y = row["mean_move_m"], row["mean_mec_radius_m"]
            axis.scatter(x, y, s=100, marker=MARKERS[policy], color=COLORS[policy])
            axis.annotate(LABELS[policy], (x, y), xytext=(6, 5),
                          textcoords="offset points", fontsize=9)
        axis.set_xlabel("平均移动距离 / m")
        axis.set_ylabel("平均观测 MEC 半径 / m")
        axis.set_title(title); axis.grid(alpha=.18)
        axis.annotate("左下更优", xy=(.06, .08), xycoords="axes fraction",
                      color="0.4", fontsize=9)
    save(fig, output)


def success_intervals(rows: list[dict], output: Path) -> None:
    scenarios = ["global_random", "edge_outward_stress"]
    fig, axis = plt.subplots(figsize=(9.4, 5.1), constrained_layout=True)
    x = np.arange(len(scenarios)); width = .24
    for pi, policy in enumerate(POLICIES):
        ps, lows, highs = [], [], []
        for scenario in scenarios:
            values = [float(r["mec_radius_m"]) <= 20 for r in rows
                      if r["scenario"] == scenario and r["policy"] == policy]
            n, k = len(values), sum(values); z = 1.96
            center = (k + z*z/2)/(n+z*z)
            half = z*math.sqrt(k*(n-k)/n + z*z/4)/(n+z*z)
            p = k/n
            ps.append(p); lows.append(p-(center-half)); highs.append((center+half)-p)
        bars = axis.bar(x+(pi-1)*width, np.array(ps)*100, width,
                        color=COLORS[policy], label=LABELS[policy],
                        yerr=np.array([lows, highs])*100, capsize=4)
        axis.bar_label(bars, labels=[f"{p*100:.1f}%" for p in ps], padding=5, fontsize=9)
    axis.set_xticks(x, ["全局随机（n=48）", "边界压力（n=16）"])
    axis.set_ylabel("MEC 半径不超过 20 m 的比例 / %")
    axis.set_ylim(0, 118); axis.set_title("20 m 阈值达标率及 Wilson 95% 区间")
    axis.grid(axis="y", alpha=.18); axis.legend(frameon=False, ncols=3)
    save(fig, output)


def radius_area(rows: list[dict], output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.0), constrained_layout=True)
    for axis, scenario, title in zip(
        axes, ("global_random", "edge_outward_stress"),
        ("(a) 全局随机", "(b) 边界压力")):
        part = [r for r in rows if r["scenario"] == scenario]
        max_r = 0.0
        for policy in POLICIES:
            selected = [r for r in part if r["policy"] == policy]
            radius = np.array([float(r["mec_radius_m"]) for r in selected])
            area = np.array([float(r["area_m2"]) for r in selected])
            max_r = max(max_r, float(radius.max()))
            axis.scatter(radius, area, s=28, alpha=.65, marker=MARKERS[policy],
                         color=COLORS[policy], label=LABELS[policy])
        rr = np.linspace(0, max_r*1.03, 200)
        axis.plot(rr, math.pi*rr**2, color="0.45", ls="--", lw=1,
                  label="圆面积 πR²（理论上界）")
        axis.set_xlabel("MEC 半径 / m"); axis.set_ylabel("后验多边形面积 / m²")
        axis.set_title(title); axis.grid(alpha=.17); axis.legend(frameon=False, fontsize=8)
    save(fig, output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path,
                        default=Path(__file__).parent / "results_mec")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    output = args.output or args.results / "visualizations"
    output.mkdir(parents=True, exist_ok=True)
    rows = load_csv(args.results / "trial_results.csv")
    candidates = load_csv(args.results / "candidate_points.csv")
    summary = json.loads((args.results / "summary.json").read_text(encoding="utf-8"))

    jobs = [
        (problem_geometry, (summary,), "01_problem_geometry.png"),
        (safe_mechanism, (), "02_safe_mechanism.png"),
        (mec_theory, (), "03_mec_theory.png"),
        (minimax_flowchart, (), "04_minimax_flowchart.png"),
        (sample_scenarios, (rows,), "05_sample_scenarios.png"),
        (strategy_actions, (rows,), "06_strategy_actions.png"),
        (candidate_landscape, (candidates,), "07_candidate_landscape.png"),
        (worst_bearing_profiles, (summary,), "08_worst_bearing_profiles.png"),
        (observed_vs_robust, (rows,), "09_observed_vs_robust.png"),
        (paired_histograms, (rows,), "10_paired_improvement_histograms.png"),
        (movement_tradeoff, (summary,), "11_movement_accuracy_tradeoff.png"),
        (success_intervals, (rows,), "12_success_rate_intervals.png"),
        (radius_area, (rows,), "13_radius_area_relationship.png"),
    ]
    for function, positional, name in jobs:
        function(*positional, output / name)
        print(name)


if __name__ == "__main__":
    main()

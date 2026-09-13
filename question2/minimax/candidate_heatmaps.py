"""Directly evaluate two second-station heatmaps; preserve the current policy."""

from __future__ import annotations

import csv
import json
from multiprocessing import Pool
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch
import numpy as np

from .mechanistic import (
    MechanisticPolicy, initial_region, reception_supports, robust_score,
)

OUT = Path(__file__).parent / "results_mec" / "candidate_heatmaps"
STEP = 25.0
CASES = [("center", "靠近圆心", np.array([0., 0.])),
         ("clipped_800m", "目标域裁切至约 800 m", np.array([1000., 0.]))]
_FIRST = None


def initialize(first):
    global _FIRST
    _FIRST = first


def evaluate(point):
    return robust_score(_FIRST, point, angle_step_deg=2., circle_sides=96).worst_mec_radius_m


def margin_grid(station, xx, yy):
    supports = reception_supports(station, 0., .05)
    points = np.column_stack((xx.ravel(), yy.ravel()))
    margin = np.empty(len(points))
    for start in range(0, len(points), 1000):
        part = points[start:start + 1000]
        margin[start:start + len(part)] = np.min(
            supports.radii - np.linalg.norm(part[:, None] - supports.points, axis=2), axis=1)
    return margin.reshape(xx.shape)


def compute():
    OUT.mkdir(parents=True, exist_ok=True)
    policy = MechanisticPolicy(circle_sides=96, angle_step_deg=2.)
    metadata = {"grid_step_m": STEP, "circle_sides": 96,
                "bearing_step_deg": 2., "safe_angle_step_deg": .05,
                "metric": "worst second-bearing MEC radius, metres",
                "notes": "Direct grid scores; no interpolated input values. Numerical angular sampling; 5 m special return omitted.",
                "cases": []}
    for name, title, station in CASES:
        cache = OUT / f"{name}.npz"
        if cache.exists():
            metadata["cases"].append(json.loads((OUT / f"{name}.json").read_text()))
            continue
        first = initial_region(station, 0., 96)
        decision = policy.decide(station, 0.)
        xs = station[0] + np.arange(-1000., 1000.1, STEP)
        ys = station[1] + np.arange(-1000., 1000.1, STEP)
        xx, yy = np.meshgrid(xs, ys)
        margin = margin_grid(station, xx, yy)
        safe = margin >= -1e-7
        points = np.column_stack((xx[safe], yy[safe]))
        print(f"{name}: evaluating {len(points)} grid points", flush=True)
        values = []
        with Pool(processes=4, initializer=initialize, initargs=(first,)) as pool:
            for i, value in enumerate(pool.imap(evaluate, points, chunksize=16), 1):
                values.append(value)
                if i % 500 == 0:
                    print(f"{name}: {i}/{len(points)}", flush=True)
        scores = np.full(xx.shape, np.nan)
        scores[safe] = values
        np.savez_compressed(cache, xs=xs, ys=ys, scores=scores, margin=margin,
                            first=first, selected=decision.point)
        with (OUT / f"{name}_grid.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["x_m", "y_m", "worst_mec_radius_m"])
            writer.writerows(zip(points[:, 0], points[:, 1], values))
        best = int(np.argmin(values))
        payload = {"name": name, "station_m": station.tolist(), "bearing_deg": 0,
                   "policy_point_m": decision.point.tolist(),
                   "policy_worst_radius_m": decision.score.worst_mec_radius_m,
                   "valid_grid_points": len(points), "grid_min_radius_m": values[best],
                   "grid_min_point_m": points[best].tolist(),
                   "grid_le20_count": int(np.sum(np.asarray(values) <= 20.))}
        (OUT / f"{name}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2))
        metadata["cases"].append(payload)
    (OUT / "summary.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    return metadata


def plot(metadata):
    """Export matching independent figures for LaTeX inclusion."""
    plt.rcParams.update({"font.family": "sans-serif",
                        "font.sans-serif": ["Noto Sans CJK SC", "Microsoft YaHei", "SimHei", "Arial Unicode MS", "Hiragino Sans GB", "DejaVu Sans"],
                        "axes.unicode_minus": False, "font.size": 11,
                        "axes.spines.top": False, "axes.spines.right": False,
                        "svg.fonttype": "path"})
    # Identical scales permit direct comparison of the independent figures.
    norm = LogNorm(vmin=10., vmax=750.)
    ticks = [10, 20, 50, 100, 200, 500, 750]
    cmap = plt.get_cmap("viridis_r").copy()
    cmap.set_bad("#edf0f3")
    for (name, title, station), info in zip(CASES, metadata["cases"]):
        data = np.load(OUT / f"{name}.npz")
        selected, first = data["selected"], data["first"]
        xs, ys, scores = data["xs"], data["ys"], data["scores"]
        fig = plt.figure(figsize=(7.8, 7.0), facecolor="white")
        ax = fig.add_axes([.115, .19, .70, .78])
        cax = fig.add_axes([.855, .19, .026, .78])
        mesh = ax.pcolormesh(xs, ys, np.ma.masked_invalid(scores), shading="nearest",
                             cmap=cmap, norm=norm, rasterized=True)
        bx = station[0] + np.linspace(-1030, 1030, 321)
        by = station[1] + np.linspace(-1030, 1030, 321)
        bxx, byy = np.meshgrid(bx, by)
        ax.contour(bx, by, margin_grid(station, bxx, byy), levels=[0],
                   colors=["#1b655a"], linewidths=1.25)
        near = max(info["policy_worst_radius_m"] * 1.05,
                   info["policy_worst_radius_m"] + .05)
        ax.contour(xs, ys, np.ma.masked_invalid(scores), levels=[near],
                   colors=["#ed3094"], linewidths=1.65)
        has_twenty = np.nanmin(scores) < 20 < np.nanmax(scores)
        if has_twenty:
            ax.contour(xs, ys, np.ma.masked_invalid(scores), levels=[20],
                       colors=["#00a3c7"], linewidths=1.6)
        ax.add_patch(Circle((0, 0), 1800, fill=False, ec="#253d56", ls="--", lw=1.5))
        ax.fill(first[:, 0], first[:, 1], color="#f59b42", alpha=.55, zorder=4)
        ax.plot(first[:, 0], first[:, 1], color="#bc590e", lw=1., zorder=4)
        ax.plot(*station, "o", color="#152b42", mec="white", mew=.9, ms=7, zorder=6)
        ax.plot(*selected, "*", color="#ff483c", mec="white", mew=.8, ms=16, zorder=7)
        ax.set(xlim=(station[0] - 1040, station[0] + 1040), ylim=(-1040, 1040),
               aspect="equal", xlabel="第二检测点 x / m", ylabel="第二检测点 y / m")
        ax.tick_params(labelsize=10)
        bar = fig.colorbar(mesh, cax=cax, ticks=ticks)
        bar.ax.set_yticklabels([str(value) for value in ticks])
        bar.set_label("最坏覆盖圆半径 J(q) / m", labelpad=8, fontsize=10)
        bar.ax.minorticks_off()
        handles = [Line2D([], [], marker="o", ls="", color="#152b42", label="第一检测点"),
                   Line2D([], [], marker="*", ls="", color="#ff483c", ms=11, label="当前策略选点"),
                   Patch(fc="#f59b42", label="第一次目标可行域"),
                   Line2D([], [], color="#1b655a", label="可靠接收域边界"),
                   Line2D([], [], color="#ed3094", label="5% 近优等值线"),
                   Line2D([], [], color="#253d56", ls="--", label="1800 m 目标圆边界")]
        if has_twenty:
            handles.append(Line2D([], [], color="#00a3c7", label="J(q) = 20 m"))
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .025),
                   ncol=3, frameon=False, fontsize=9, handlelength=1.8, columnspacing=1.2)
        for extension in ("png", "svg"):
            fig.savefig(OUT / f"{name}_heatmap.{extension}", dpi=320, facecolor="white")
        plt.close(fig)

    (OUT / "latex_example.tex").write_text(r"""% Use XeLaTeX / LuaLaTeX with a Chinese-capable document class.
% Preamble: \usepackage{graphicx}, \usepackage{subcaption}, \usepackage{placeins}
% Put the two PNG files alongside the main .tex, or set \graphicspath.
\begin{figure}[htbp]
  \centering
  \begin{subfigure}[t]{0.48\textwidth}
    \centering
    \includegraphics[width=\linewidth]{center_heatmap.png}
    \caption{第一检测点位于圆心，前向可行距离约为1500米。}
    \label{fig:q2-center}
  \end{subfigure}
  \hfill
  \begin{subfigure}[t]{0.48\textwidth}
    \centering
    \includegraphics[width=\linewidth]{clipped_800m_heatmap.png}
    \caption{第一检测点为$(1000,0)$，前向可行距离被裁切至约800米。}
    \label{fig:q2-clipped}
  \end{subfigure}
  \caption{第二检测点候选区域的最坏覆盖圆半径热力图。两种情形的第一次示向度均为$0^\circ$，使用相同对数色标。红星表示当前策略输出，粉色等值线对应相对该策略值的5\%近优阈值。热力值采用25米网格逐点评分；圆域采用96边外接多边形，未来示向度按2度网格并补充关键角采样，接收约束角步长为0.05度。结果为数值近似，沿用忽略5米特殊返回的主模型。}
  \label{fig:q2-heatmaps}
\end{figure}

\FloatBarrier

由图~\ref{fig:q2-heatmaps}可见，第二检测点的选择由目标可行域的距离范围、
测向交会几何和可靠接收条件共同决定。沿第一次示向轴移动难以有效消除纵向位置
不确定性，因而较优检测点通常具有明显的横向位移。目标圆边界的裁切不仅缩短
目标可行域，还可能扩大可靠接收域，使策略能够以较短的移动距离形成有效交会。
在本文两个对称案例中，较优候选区域均呈上下镜像分布；当目标可行域的前向距离
由约 $1500\,\mathrm{m}$ 裁切至约 $800\,\mathrm{m}$ 时，当前策略的最坏覆盖半径
由 $55.84\,\mathrm{m}$ 降至 $20.59\,\mathrm{m}$。
""", encoding="utf-8")


if __name__ == "__main__":
    payload = compute()
    plot(payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)

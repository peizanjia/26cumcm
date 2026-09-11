"""Reproducible standalone scientific figures from saved experiment results."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D

ROOT = Path(__file__).parent / "results"
PALETTE = ["#1767a0", "#b37a18", "#ba5c39", "#727d35"]


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.labelcolor": "#30343b", "text.color": "#30343b",
                         "figure.facecolor": "white", "savefig.facecolor": "white"})
    report = json.loads((ROOT/"evaluation.json").read_text())
    training = list(csv.DictReader((ROOT/"training.csv").open()))
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.8), layout="constrained")
    for i in range(4):
        rows = [r for r in training if int(r["model"]) == i]
        axes[0].plot([int(r["step"]) for r in rows], [float(r["mean_area_m2"]) for r in rows],
                     label=f"Network {i+1}", color=PALETTE[i], lw=1.8, ls=["-", "--", ":", "-."][i])
    axes[0].set(title="Four independent networks", xlabel="Training step", ylabel="Validation mean residual area (m²)")
    axes[0].legend(frameon=False, ncol=2)
    axes[0].grid(alpha=.18)
    names = ["neural", "neural_certified", "analytic_moments", "adaptive_60_28", "fixed_750_375"]
    labels = ["Neural policy", "Neural + reception certificate", "Analytical posterior-moment rule", "Adaptive geometric baseline", "Fixed forward 750 / lateral 375 m"]
    rows = [{r["policy"]: r for r in report["summary"]}[name] for name in names]
    values = [r["mean_area_m2"] for r in rows]
    errors = [r["ci95_high"]-r["mean_area_m2"] for r in rows]
    y = np.arange(len(rows))
    axes[1].barh(y, values, xerr=errors, capsize=3, color=[PALETTE[0], "#658bab", "#939da7", "#adb5bd", "#cbd0d5"])
    axes[1].set_yticks(y, labels)
    axes[1].invert_yaxis()
    axes[1].set_xlim(0, max(values)*1.15)
    for j, value in enumerate(values):
        axes[1].text(value+10, j, f"{value:.1f}", va="center")
    axes[1].set(title="Independent held-out comparison", xlabel="Mean residual area (m²); lower is better")
    fig.suptitle("Question 2 | Expected-area policy learning", fontsize=15, fontweight="bold")
    fig.supxlabel("Synthetic model: 8,192 held-out states × 256 posterior draws; bars show state-cluster 95% CIs.", fontsize=9)
    for ext in ["png", "svg"]:
        fig.savefig(ROOT/f"training_comparison.{ext}", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 10), layout="constrained")
    tags = ["origin_east", "edge_inward", "edge_outward", "oblique"]
    titles = ["S₁=(0,0), θ₁=0°", "S₁=(1400,0), θ₁=180°",
              "S₁=(1500,0), θ₁=0°  [zoomed axes]", "S₁=(800,1000), θ₁=225°"]
    for ax, tag, title in zip(axes.ravel(), tags, titles):
        d = np.load(ROOT/"candidates"/f"{tag}.npz")
        mesh = ax.pcolormesh(d["xx"], d["yy"], np.maximum(d["scores"], 1),
                             norm=LogNorm(10, 40000), cmap="cividis", shading="nearest", rasterized=True)
        ax.contour(d["xx"], d["yy"], d["candidate"].astype(float), [.5], colors=["#f8fafc"], linewidths=1.7)
        ax.contour(d["xx"], d["yy"], d["robust"].astype(float), [.5], colors=["#edaa43"], linestyles="--", linewidths=1.7)
        poly = d["polygon"]
        pp = np.vstack([poly, poly[:1]])
        ax.plot(pp[:, 0], pp[:, 1], color="#111827", lw=1.3)
        ax.scatter(*d["nn"], marker="*", s=130, c="white", edgecolors="black", linewidths=.8, zorder=6)
        ax.scatter(0, 0, marker="o", s=25, c="black", zorder=6)
        ax.set(title=title, xlabel="Forward displacement along first bearing (m)", ylabel="Lateral displacement (m)")
        ax.set_aspect("equal")
        ax.set_xlim(-100, 1600)
        ax.set_ylim(-700, 700)
        if tag == "edge_outward":
            ax.set_xlim(100, 400)
            ax.set_ylim(-150, 150)
    fig.colorbar(mesh, ax=axes, label="Expected residual area (m²), logarithmic color scale", shrink=.8, pad=.015)
    handles = [Line2D([], [], color="#737373", lw=1.7, label="5% MC sublevel boundary"),
               Line2D([], [], color="#edaa43", lw=1.7, ls="--", label="1000 m reception certificate"),
               Line2D([], [], marker="*", color="none", markerfacecolor="white", markeredgecolor="black", markersize=11, label="Raw neural action")]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False)
    fig.suptitle("Second-point candidate regions in bearing-aligned coordinates\n20 m grid; 2,048 conditional samples per state; grid/MC estimates", fontsize=14)
    for ext in ["png", "svg"]:
        fig.savefig(ROOT/f"candidate_regions.{ext}", dpi=170)
    plt.close(fig)
    print("Saved training_comparison and candidate_regions as PNG/SVG")


if __name__ == "__main__":
    main()

"""Scientific figures backed by saved fixed-truth and policy diagnostics."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

OUT=Path(__file__).parents[1]/"results"/"symmetry"


def save(fig,name):
    for ext in ("png","svg"):fig.savefig(OUT/f"{name}.{ext}",dpi=160)
    plt.close(fig)


def main():
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,
                         "axes.spines.top":False,"axes.spines.right":False,
                         "figure.facecolor":"white","savefig.facecolor":"white"})
    cases=json.loads((OUT/"fixed_truth"/"cases.json").read_text())
    fig,axes=plt.subplots(2,3,figsize=(16,10),layout="constrained")
    for ax,case in zip(axes.ravel(),cases):
        d=np.load(OUT/"fixed_truth"/(case["tag"]+".npz"))
        mesh=ax.pcolormesh(d["xx"],d["yy"],np.maximum(d["score"],1),cmap="cividis",norm=LogNorm(1,40000),shading="nearest",rasterized=True)
        if d["no_signal"].any():
            ax.contourf(d["xx"],d["yy"],d["no_signal"],[.5,1.5],colors=["#b6b9bd"],alpha=.85)
        poly=np.vstack([d["polygon"],d["polygon"][:1]])
        ax.plot(poly[:,0],poly[:,1],color="#111827",lw=.9)
        ax.scatter(0,0,c="black",s=18,zorder=5)
        ax.scatter(*d["target"],marker="X",c="#ef9030",edgecolors="black",s=90,lw=.7,zorder=10)
        ax.scatter(d["new"][:,0],d["new"][:,1],marker="*",c="white",edgecolors="black",s=130,lw=.8,zorder=9)
        ax.scatter(*d["old"],facecolors="none",edgecolors="#eaf4ff",marker="o",s=85,lw=1.3,zorder=8)
        ax.set(title=f"S={tuple(case['s'])}; G=({case['g'][0]:.0f},{case['g'][1]:.0f})\nFirst bearing error = {case['first_error_deg']:.1f}°",
               xlabel="Forward displacement (m)",ylabel="Lateral displacement (m)")
        ax.set_aspect("equal")
    fig.colorbar(mesh,ax=axes,shrink=.85,pad=.012,label="E[residual area | fixed S,G,first bearing,R] (m²); log scale")
    fig.legend(handles=[Line2D([],[],marker="X",ls="",color="#ef9030",markeredgecolor="black",label="Fixed true target"),
                        Line2D([],[],marker="*",ls="",markerfacecolor="white",markeredgecolor="black",markersize=12,label="New NN: right and left"),
                        Line2D([],[],marker="o",ls="",markerfacecolor="none",markeredgecolor="#777",label="Original NN"),
                        Patch(facecolor="#b6b9bd",label="No signal: retain first area")],loc="outside lower center",ncol=4,frameon=False)
    fig.suptitle("Fixed-truth second-point score landscapes\nR=1250 m; first reading fixed; average over 256 second-error midpoints; near + optical gives zero",fontsize=14)
    save(fig,"fixed_truth_heatmaps")
    d=np.load(OUT/"boundary_curve.npz")
    fig,axes=plt.subplots(1,3,figsize=(17,4.9),layout="constrained")
    for ax,column,title in [(axes[0],0,"Forward component"),(axes[1],1,"Lateral magnitude")]:
        for key,label,color,ls in [("old","Original NN","#9a6a18","--"),("new","Boundary-scaled shared NN","#1767a0","-"),("approximate","Posterior-moment formula","#343a40",":")]:
            ax.plot(d["length_m"],abs(d[key][:,column]),label=label,color=color,ls=ls,lw=2)
        ax.set(title=title,xlabel="Available central-ray length L (m)",ylabel="Displacement (m)")
        ax.set_xlim(0,1500);ax.set_ylim(bottom=0);ax.grid(alpha=.2)
    axes[0].legend(frameon=False,fontsize=9)
    land=np.load(OUT/"fixed_landscape.npz");fixed=json.loads((OUT/"optimized_fixed.json").read_text())
    mesh=axes[2].pcolormesh(land["xx"],land["yy"],land["score"],cmap="cividis",shading="nearest",rasterized=True)
    axes[2].contour(land["xx"],land["yy"],land["score"],levels=[550,575,600,650,750],colors="white",linewidths=.8)
    axes[2].scatter(fixed["a_m"],fixed["abs_b_m"],marker="*",s=130,c="white",edgecolors="black",label="Optimized on train/validation")
    axes[2].scatter(750,375,marker="x",s=70,c="black",label="Unoptimized 750 / 375")
    axes[2].set(title="Constant-policy objective landscape",xlabel="Fixed forward a (m)",ylabel="Fixed lateral magnitude |b| (m)")
    axes[2].legend(frameon=False,fontsize=8,loc="lower left")
    fig.colorbar(mesh,ax=axes[2],label="Expected area (m²)",shrink=.8)
    fig.suptitle("Explicit policy relationships and a fair fixed-step baseline",fontsize=14)
    fig.supxlabel("Left/middle: S=(1800-L,0), first bearing east. Right: separate optimization split, 512 states × 64 draws.",fontsize=9)
    save(fig,"policy_relationships")
    print("Saved fixed_truth_heatmaps and policy_relationships (PNG/SVG)")


if __name__=="__main__":main()

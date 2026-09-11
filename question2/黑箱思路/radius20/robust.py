"""Conservative all-bearing angular-bin bounds on an outer first polygon.

Analytical covering argument, floating-point implementation, no interval proof.
Only selected candidate points are certified by the model bound, not grid cells.
"""
import json
from pathlib import Path
import numpy as np
from numba import njit,prange
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from question2.最小包围圆思路.core import mec
from ..geometry import clip_cpu,DELTA,first_polygon
from ..symmetry.fixed_truth import CASES

ROOT=Path(__file__).parent


@njit(cache=True,parallel=True)
def conservative_bounds(poly,qs,bins=720):
    out=np.empty(len(qs));r0=mec(poly)
    halfwidth=DELTA+np.pi/bins
    for i in prange(len(qs)):
        q=qs[i];maxdist=0.
        for v in poly:maxdist=max(maxdist,np.linalg.norm(v-q))
        if maxdist>1000-1e-6 or np.linalg.norm(q)<1e-9 or r0<=19.95:
            out[i]=r0;continue
        worst=0.
        for k in range(bins):
            angle=(k+.5)*2*np.pi/bins
            clipped=poly.copy()
            for sign in (-1,1):
                a=angle+sign*halfwidth
                n=np.array([-sign*np.sin(a),sign*np.cos(a)])
                clipped=clip_cpu(clipped,n,np.dot(n,q))
            worst=max(worst,mec(clipped))
            if worst>19.95:
                # Universal bound r0 remains valid; no claim this point is infeasible.
                worst=r0;break
        out[i]=min(r0,worst)
    return out


def main():
    output=ROOT/'results';records=[]
    fig,axes=plt.subplots(1,3,figsize=(15,5),layout='constrained')
    for ax,case in zip(axes,[CASES[0],CASES[3],CASES[4]]):
        data=np.load(output/f'maps/posterior_{case["tag"]}.npz')
        xx=data['xx'][::4,::4];yy=data['yy'][::4,::4]
        q=np.column_stack([xx.ravel(),yy.ravel()]);s=np.array(case['s'])
        poly=first_polygon(s/1500,1024)*1500
        bounds=conservative_bounds(poly,q-s)
        mask=bounds<=19.95
        np.savez_compressed(output/f'maps/conservative_{case["tag"]}.npz',xx=xx,yy=yy,bound_m=bounds,qualified=mask)
        record=dict(case=case['tag'],tested_candidates=len(q),qualified_candidates=int(mask.sum()),
            angular_bins=720,angular_bin_width_deg=.5,expanded_halfwidth_deg=1.25,safety_margin_m=.05,
            scope='Outer polygon + all-angle containment bound; floating point, no interval certificate; points only, not cells')
        if mask.any():
            ids=np.flatnonzero(mask);k=ids[np.argmin(np.linalg.norm(q[ids]-s,axis=1))]
            record.update(nearest_qualified_q=q[k].tolist(),bound_m=float(bounds[k]))
        records.append(record)
        ax.scatter(q[:,0],q[:,1],s=3,color='#dedede',label='Not established')
        ax.scatter(q[mask,0],q[mask,1],s=9,color='#2171b5',label='Conservative bound <=19.95 m')
        ax.scatter(*s,c='black',s=25)
        ax.set(title=f'{case["tag"]}: {mask.sum()} candidate points',xlabel='Global x (m)',ylabel='Global y (m)',aspect='equal')
        print(json.dumps(record),flush=True)
    axes[1].legend(fontsize=8,loc='upper right')
    fig.suptitle('Conservative geometric inner candidates | no true target used\nGray means unestablished, not infeasible; floating-point bounds, not continuous-region certificates',fontsize=12)
    fig.savefig(output/'conservative_candidates.png',dpi=180);plt.close(fig)
    (output/'conservative_regions.json').write_text(json.dumps(records,indent=2))


if __name__=='__main__':main()

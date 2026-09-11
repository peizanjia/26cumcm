"""Fixed-truth and posterior radius landscapes, linear 0..25 m colors."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .core import evaluate
from question2.黑箱思路.geometry import first_polygon,DELTA
from question2.黑箱思路.environment import sample_posterior
from question2.黑箱思路.symmetry.fixed_truth import CASES

ROOT=Path(__file__).parent


def events(case,posterior,n=256,seed=30260911):
    p=np.array(case['s'],dtype=float)/1500
    poly=first_polygon(p,1024)*1500
    rng=np.random.default_rng(seed)
    if posterior:
        g=sample_posterior(p[None],n,seed)[0]*1500
        low=np.maximum(1000,np.linalg.norm(g,axis=1))
        r=low+(1500-low)*rng.random(n)
        e=rng.uniform(-DELTA,DELTA,n)
    else:
        g=np.broadcast_to(np.array(case['g'])-np.array(case['s']),(n,2)).copy()
        r=np.full(n,1250.)
        # Include endpoints in the diagnostic sample maximum, never claim a
        # continuous worst-case certificate between angle nodes.
        e=np.linspace(-DELTA,DELTA,n)
    return dict(polygons=poly[None],counts=np.array([len(poly)]),targets=g[None],
                errors=e[None],radii=r[None])


def grid(d,q):
    rows=[]
    for start in range(0,len(q),128):
        qq=q[start:start+128]
        dd={k:np.repeat(v,len(qq),axis=0) for k,v in d.items()}
        r=evaluate(dd,qq)
        rows.append(np.column_stack([r.mean(1),r.min(1),r.max(1),r.var(1),r.std(1),
            np.median(r,axis=1),np.quantile(r,.95,axis=1),(r<=20).mean(1)]))
    return np.concatenate(rows)


def main():
    output=ROOT/'landscapes';output.mkdir(exist_ok=True)
    records=[]
    for posterior in (False,True):
        cases=CASES if not posterior else [CASES[0],CASES[3],CASES[4]]
        kind='posterior' if posterior else 'fixed_truth'
        fig,axes=plt.subplots(2,3,figsize=(16,9),layout='constrained') if not posterior else plt.subplots(1,3,figsize=(16,4.8),layout='constrained')
        figp,axp=plt.subplots(2,3,figsize=(16,9),layout='constrained') if not posterior else plt.subplots(1,3,figsize=(16,4.8),layout='constrained')
        axes=np.asarray(axes).ravel();axp=np.asarray(axp).ravel()
        cmap=plt.get_cmap('viridis_r').copy();cmap.set_bad('#eeeeee')
        for i,case in enumerate(cases):
            short=case['tag'] in ('edge_out','oblique_boundary')
            xx,yy=np.meshgrid(np.linspace(-50,600,131) if short else np.linspace(-100,1600,171),
                              np.linspace(-300,300,121) if short else np.linspace(-750,750,151))
            q=np.column_stack([xx.ravel(),yy.ravel()])
            d=events(case,posterior)
            z=grid(d,q);mean=z[:,0].reshape(xx.shape);mx=z[:,2].reshape(xx.shape);prob=z[:,7].reshape(xx.shape)
            dist=np.linalg.norm(q,axis=1)
            masks=[z[:,0]<=20,z[:,7]>=.95,z[:,2]<=20]
            best={}
            for name,mask in zip(('expected_le20','chance95','samplemax_le20'),masks):
                idx=np.flatnonzero(mask)
                if len(idx):
                    k=idx[np.argmin(dist[idx])]
                    best[name]=dict(q_m=q[k].tolist(),distance_m=float(dist[k]),mean_m=float(z[k,0]),
                        sample_max_m=float(z[k,2]),success=float(z[k,7]),grid_points=int(len(idx)))
                    if posterior:
                        validation=events(case,True,n=4096,seed=31260911+i)
                        rr=evaluate(validation,q[k:k+1])[0]
                        best[name]['independent_validation']=dict(mean=float(rr.mean()),max=float(rr.max()),success=float((rr<=20).mean()))
                else:best[name]=dict(grid_points=0,q_m=None)
            records.append(dict(kind=kind,case=case['tag'],samples=256,first=case['s'],
                target=case['g'] if not posterior else None,first_theta_deg=0,
                hidden_R_m=1250 if not posterior else 'conditional uniform',nearest_grid=best))
            np.savez_compressed(output/f'{kind}_{case["tag"]}.npz',xx=xx,yy=yy,metrics=z,
                feasible_mean=masks[0].reshape(xx.shape),feasible_chance95=masks[1].reshape(xx.shape),
                feasible_samplemax=masks[2].reshape(xx.shape))
            with (output/f'{kind}_{case["tag"]}.csv').open('w',newline='') as f:
                w=csv.writer(f);w.writerow(['along_m','lateral_m','mean_m','min_m','sample_max_m','variance_m2','std_m','median_m','p95_m','p_le20','distance_m','mean_le20','chance95','samplemax_le20'])
                w.writerows(np.column_stack([q,z,dist,*masks]))
            ax=axes[i];im=ax.pcolormesh(xx,yy,np.ma.masked_where(mean>25,mean),cmap=cmap,vmin=0,vmax=25,shading='auto',rasterized=True)
            if mean.min()<20<mean.max():ax.contour(xx,yy,mean,levels=[20],colors='black',linewidths=1.4)
            if mx.min()<20<mx.max():ax.contour(xx,yy,mx,levels=[20],colors='#d73027',linestyles='--',linewidths=1.2)
            ip=axp[i].pcolormesh(xx,yy,prob,cmap='viridis',vmin=0,vmax=1,shading='auto',rasterized=True)
            if prob.min()<.95<prob.max():axp[i].contour(xx,yy,prob,levels=[.95],colors='red',linewidths=1.4)
            for a in (ax,axp[i]):
                a.scatter([0],[0],s=35,c='black',marker='o',zorder=5)
                if not posterior:
                    g=np.array(case['g'])-np.array(case['s']);a.scatter(*g,c='#ff6600',s=60,marker='x',zorder=5)
                a.set(xlabel='Forward displacement (m)',ylabel='Lateral displacement (m)',title=case['tag'],aspect='equal')
            print(kind,case['tag'],'mean minimum',float(mean.min()),'feasible',best,flush=True)
        fig.colorbar(im,ax=axes.tolist(),label='Expected MEC radius (m); gray >25',ticks=[0,5,10,15,20,25],shrink=.8)
        figp.colorbar(ip,ax=axp.tolist(),label='Estimated P(radius <=20 m)',shrink=.8)
        fig.suptitle(kind+' | black: mean=20; red dashed: sampled max=20\n'+
            ('Unknown target: posterior events; sampled feasibility is not a guarantee' if posterior else 'Fixed S, G, bearing, R=1250 m; orange x = hidden target; optical near=0'),fontsize=13)
        figp.suptitle(kind+' | red: estimated 95% success contour; 256 events',fontsize=13)
        fig.savefig(ROOT/f'{kind}_radius.png',dpi=160);fig.savefig(ROOT/f'{kind}_radius.svg')
        figp.savefig(ROOT/f'{kind}_probability.png',dpi=160)
        plt.close(fig);plt.close(figp)
    (ROOT/'feasible_regions.json').write_text(json.dumps(records,indent=2))


if __name__=='__main__':main()

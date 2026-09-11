"""Global R1800 and emphasized local radius/probability maps with policy overlays."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm,ListedColormap
from matplotlib.patches import Circle,Rectangle
from question2.最小包围圆思路.landscapes import events
from question2.最小包围圆思路.core import evaluate
from ..symmetry.fixed_truth import CASES
from .policy import load_policy

ROOT=Path(__file__).parent


def grid(d,q):
    pieces=[]
    for start in range(0,len(q),128):
        qq=q[start:start+128];dd={k:np.repeat(v,len(qq),axis=0) for k,v in d.items()}
        r=evaluate(dd,qq)
        pieces.append(np.column_stack([r.mean(1),r.std(1),np.median(r,axis=1),
            np.quantile(r,.95,axis=1),r.min(1),r.max(1),(r<=20).mean(1),
            (r<=50).mean(1),np.abs(r-20).mean(1)]))
    return np.concatenate(pieces)


def contour(ax,x,y,z,level,**kwargs):
    if np.nanmin(z)<level<np.nanmax(z):ax.contour(x,y,z,levels=[level],**kwargs)


def main():
    output=ROOT/'results/maps';output.mkdir(exist_ok=True)
    policies={n:load_policy(n=='orthogonal') for n in ('plain','orthogonal')}
    cmap=ListedColormap(['#f7fbff','#deebf7','#c6dbef','#9ecae1','#6baed6','#4292c6','#2171b5','#084594'])
    cmap.set_over('#d5d5d5');norm=BoundaryNorm([0,5,10,15,20,25,30,40,50],cmap.N)
    records=[]
    for posterior in (False,True):
        kind='posterior' if posterior else 'fixed_truth'
        for case in (CASES if not posterior else [CASES[0],CASES[3],CASES[4]]):
            s=np.array(case['s'],float);g=np.array(case['g'],float)
            opts={n:np.array(list(pol.options(*s,0).values())) for n,pol in policies.items()}
            short=case['tag'] in ('edge_out','oblique_boundary')
            if short:bounds=[s[0]-100,s[0]+600,s[1]-350,s[1]+350]
            else:bounds=[s[0]+100,s[0]+1500,s[1]-700,s[1]+700]
            global_x,global_y=np.meshgrid(np.linspace(-1850,1850,101),np.linspace(-1850,1850,101))
            xx,yy=np.meshgrid(np.linspace(bounds[0],bounds[1],201),np.linspace(bounds[2],bounds[3],201))
            d=events(case,posterior,n=256,seed=37260911)
            zg=grid(d,np.column_stack([global_x.ravel(),global_y.ravel()])-s)
            q=np.column_stack([xx.ravel(),yy.ravel()]);z=grid(d,q-s)
            shape=xx.shape;mean=z[:,0].reshape(shape);prob=z[:,6].reshape(shape);maximum=z[:,5].reshape(shape)
            masks=[mean<=20,prob>=.95,maximum<=20]
            selections={}
            for label,mask in zip(('mean_le20','chance95','samplemax_le20'),masks):
                ids=np.flatnonzero(mask.ravel())
                record=dict(grid_points=int(len(ids)),q=None)
                if len(ids):
                    k=int(ids[np.argmin(z[ids,8])])
                    record.update(q=q[k].tolist(),mae20=float(z[k,8]),mean=float(z[k,0]),p_le20=float(z[k,6]))
                    if posterior:
                        independent=events(case,True,n=4096,seed=38260911)
                        rr=evaluate(independent,(q[k]-s)[None])[0]
                        record['independent']=dict(mean=float(rr.mean()),p_le20=float((rr<=20).mean()),
                            max=float(rr.max()),mae20=float(np.abs(rr-20).mean()),
                            passes_selected_rule=bool(rr.mean()<=20 if label=='mean_le20' else
                                ((rr<=20).mean()>=.95 if label=='chance95' else rr.max()<=20)))
                selections[label]=record
            meta=dict(kind=kind,case=case['tag'],first=s.tolist(),target=None if posterior else g.tolist(),
                samples=256,bounds=bounds,global_step_m=37,local_step_m=(bounds[1]-bounds[0])/200,
                selected=selections,policies={k:v.tolist() for k,v in opts.items()},
                metric_columns=['mean_m','std_m','median_m','p95_m','min_m','samplemax_m','p_le20','p_le50','mae20_m'])
            records.append(meta)
            np.savez_compressed(output/f'{kind}_{case["tag"]}.npz',xx=xx,yy=yy,metrics=z,
                global_x=global_x,global_y=global_y,global_metrics=zg,
                mean_feasible=masks[0],chance95_feasible=masks[1],samplemax_feasible=masks[2])
            np.savetxt(output/f'{kind}_{case["tag"]}_local.csv',np.column_stack([q,z]),delimiter=',',
                header='x_global_m,y_global_m,'+','.join(meta['metric_columns']),comments='')
            fig,axes=plt.subplots(1,3,figsize=(19,6.6),gridspec_kw={'width_ratios':[1,1.65,1.65]},layout='constrained')
            im=axes[0].pcolormesh(global_x,global_y,zg[:,0].reshape(global_x.shape),cmap=cmap,norm=norm,shading='auto',rasterized=True)
            axes[0].add_patch(Rectangle((bounds[0],bounds[2]),bounds[1]-bounds[0],bounds[3]-bounds[2],fill=False,ec='#c0398f',lw=1.4))
            axes[0].set(xlim=(-1850,1850),ylim=(-1850,1850),title='Global view | R = 1800 m')
            axes[1].pcolormesh(xx,yy,mean,cmap=cmap,norm=norm,shading='auto',rasterized=True)
            contour(axes[1],xx,yy,mean,20,colors='black',linewidths=1.4)
            contour(axes[1],xx,yy,mean,50,colors='black',linestyles='--',linewidths=.8)
            axes[1].set(title='Local | expected enclosing radius',xlim=bounds[:2],ylim=bounds[2:])
            ip=axes[2].pcolormesh(xx,yy,prob,vmin=0,vmax=1,cmap='YlGnBu',shading='auto',rasterized=True)
            contour(axes[2],xx,yy,prob,.95,colors='#c0398f',linewidths=1.6)
            contour(axes[2],xx,yy,maximum,20,colors='black',linestyles='--',linewidths=1.2)
            axes[2].set(title='Local | P(radius <= 20 m)',xlim=bounds[:2],ylim=bounds[2:])
            if selections['chance95']['q'] is not None:
                a=selections['chance95']['q'];axes[2].scatter(*a,marker='D',s=65,c='#c0398f',edgecolor='white',zorder=7,label='Grid feasible strategy')
            for ax in axes:
                ax.add_patch(Circle((0,0),1800,fill=False,ec='#333333',lw=1.1))
                ax.scatter(*s,c='black',s=32,label='First monitor',zorder=6)
                if not posterior:ax.scatter(*g,c='#e67e22',marker='x',s=60,linewidths=2,label='Hidden truth',zorder=6)
                for name,marker,color in [('plain','o','#8e44ad'),('orthogonal','*','#c0392b')]:
                    pts=opts[name];ax.scatter(pts[:,0],pts[:,1],marker=marker,c=color,s=65 if marker=='o' else 125,
                        edgecolor='white',linewidth=.7,zorder=6,label=name+' NN (both sides)')
                ax.set(xlabel='Global x (m)',ylabel='Global y (m)',aspect='equal')
                ax.tick_params(labelsize=9)
            axes[2].legend(loc='upper right',fontsize=8,framealpha=.85)
            fig.colorbar(im,ax=axes[:2],label='Mean MEC radius (m); gray > 50',ticks=[0,5,10,15,20,25,30,40,50],extend='max',shrink=.68)
            fig.colorbar(ip,ax=axes[2],label='Estimated probability',ticks=[0,.25,.5,.75,.95,1],shrink=.68)
            fig.suptitle(f'{kind} / {case["tag"]} | 256 events per candidate; first bearing = east\n'+
                ('Unknown target: posterior integral. Magenta = 95%; dashed black = sampled max radius 20, not a guarantee.' if posterior else
                 'Fixed target and hidden R=1250 m; diagnostics only. Black local contour = mean radius 20 m.'),fontsize=12)
            fig.savefig(output/f'{kind}_{case["tag"]}.png',dpi=160)
            fig.savefig(output/f'{kind}_{case["tag"]}.svg')
            plt.close(fig)
            print(kind,case['tag'],json.dumps(selections),flush=True)
    (ROOT/'results/feasible_regions.json').write_text(json.dumps(records,indent=2))


if __name__=='__main__':main()

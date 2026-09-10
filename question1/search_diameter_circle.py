"""Monte Carlo search for diameter-circle failures in valid bearing localization.

Run python -m question1.search_diameter_circle --trials 2500
Each scenario generates one source and a nested sequence of monitoring points.
Both noise models share geometry. No official simulator is contacted.
"""

import argparse
from collections import defaultdict
import csv
import gzip
import json
from importlib.metadata import version
from pathlib import Path
import platform
import time

import numpy as np

from .enclosing_circle import diameter_circle_metrics
from .geometry import bearing_halfplanes, intersect_bearings
from .simulate import BearingEnvironment

DEFAULT_COUNTS = (2, 3, 4, 6, 8, 12, 20, 32)
ROW_FIELDS = ['seed','noise','layout','n','status','vertices_count','diameter_m',
              'area_m2','circularity','radius_ratio','mec_radius_m',
              'max_outside_m','failure','source_outside','source_outside_m',
              'tolerance_m','polygon_inside_arena']


def monitoring_sequence(env, count, layout):
    if layout == 'uniform_area':
        return env.sample_monitoring_points(count)
    if layout != 'local_walk':
        raise ValueError('unknown layout')
    # A correlated walk through visible locations; this models measurement
    # geometry, not an implementable unknown-source search policy.
    points = [env.sample_monitoring_points(1)[0]]
    heading = env.rng.uniform(0,2*np.pi)
    for _ in range(count-1):
        for attempt in range(10000):
            heading = (heading + env.rng.normal(0,0.7) if attempt == 0
                       else env.rng.uniform(0,2*np.pi))
            step = env.rng.uniform(50,250)
            candidate = points[-1] + step*np.array([np.cos(heading),np.sin(heading)])
            distance = np.linalg.norm(candidate-env.source)
            if np.linalg.norm(candidate) <= env.radius and 50 <= distance <= env.receive_radius:
                points.append(candidate)
                break
        else:
            raise RuntimeError('Cannot generate a valid walk')
    return np.array(points)


def wilson(k,n):
    if not n:
        return None,None
    z=1.959963984540054
    p=k/n
    mid=(p+z*z/(2*n))/(1+z*z/n)
    half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return max(0.,mid-half),min(1.,mid+half)


def aggregate(rows,keys,confidence=False):
    groups=defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in keys)].append(row)
    summaries=[]
    for key,group in sorted(groups.items()):
        valid=[r for r in group if r['status']=='bounded']
        out=dict(zip(keys,key))
        out.update(total=len(group),bounded=len(valid),
                   unbounded=sum(r['status']=='unbounded' for r in group),
                   degenerate=sum(r['status']=='degenerate' for r in group))
        if valid:
            failures=sum(r['failure'] for r in valid)
            out.update(failures=failures,failure_rate=failures/len(valid),
                       source_outside_count=sum(r['source_outside'] for r in valid),
                       median_vertices=float(np.median([r['vertices_count'] for r in valid])),
                       mean_circularity=float(np.mean([r['circularity'] for r in valid])),
                       median_circularity=float(np.median([r['circularity'] for r in valid])),
                       median_ratio=float(np.median([r['radius_ratio'] for r in valid])),
                       p95_ratio=float(np.percentile([r['radius_ratio'] for r in valid],95)),
                       max_ratio=max(r['radius_ratio'] for r in valid),
                       failure_over_1pct=sum(r['radius_ratio']>1.01 for r in valid)/len(valid),
                       polygon_inside_arena_count=sum(r['polygon_inside_arena'] for r in valid))
            if confidence:
                out['ci_low'],out['ci_high']=wilson(failures,len(valid))
        summaries.append(out)
    return summaries


def write_csv(path,rows):
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def symmetric_counterexample():
    """Controlled valid measurements giving an equilateral feasible triangle.

    This is a constructed example, not a random draw or an official reading.
    All three bearings have a fixed +0.8 degree error within the allowed bound.
    """
    azimuth=np.array([0.,120.,240.])
    angle=np.deg2rad(azimuth)
    points=1000*np.column_stack((np.cos(angle),np.sin(angle)))
    true=(azimuth+180)%360
    bearings=(true+.8)%360
    result=intersect_bearings(points,bearings)
    assert result.status=='bounded' and len(result.vertices)==3
    readings=[{'measure_result':'direction','svd_deg':float(b),'true_deg':float(t),
               'error_deg':.8,'distance_m':1000.} for b,t in zip(bearings,true)]
    return {'seed':'constructed','noise':'fixed +0.8 deg','layout':'symmetric construction',
            'n':3,'arena_radius_m':1800.,'receive_radius_m':1200.,'source':[0.,0.],
            'points':points.tolist(),'measurements':readings,
            'vertices':result.vertices.tolist(),'polygon_inside_arena':True,
            'metrics':diameter_circle_metrics(result.vertices,[0,0])}


def run_study(trials,seed,counts,noises,layouts,output):
    output.mkdir(parents=True,exist_ok=True)
    rows,examples,transitions=[],{},[]
    start=time.perf_counter()
    for layout in layouts:
        for noise in noises:
            for trial in range(trials):
                env=BearingEnvironment(seed=seed+trial,noise=noise)
                points=monitoring_sequence(env,max(counts),layout)
                measurements=[env.measure(p) for p in points]
                assert all(m['measure_result']=='direction' for m in measurements)
                bearings=np.array([m['svd_deg'] for m in measurements])
                previous=None
                for n in counts:
                    result=intersect_bearings(points[:n],bearings[:n])
                    row={'seed':env.seed,'noise':noise,'layout':layout,'n':n,'status':result.status}
                    if result.status=='empty':
                        raise AssertionError('Valid bounded-error observations produced an empty set')
                    if result.status=='bounded':
                        poly=result.vertices
                        a,b=bearing_halfplanes(points[:n],bearings[:n])
                        if np.max(a@env.source-b)>1e-6 or np.max(a@poly.T-b[:,None])>1e-5:
                            raise AssertionError('Source or intersection violates bearing constraints')
                        if len(poly)<3:
                            row['status']='degenerate'
                        else:
                            metrics=diameter_circle_metrics(poly,env.source)
                            if not 1-1e-8 <= metrics['radius_ratio'] <= 2/np.sqrt(3)+1e-7:
                                raise AssertionError('Circle ratio violates the planar geometric bound')
                            row.update({k:metrics[k] for k in ROW_FIELDS if k in metrics})
                            row['polygon_inside_arena']=bool(np.max(np.linalg.norm(poly,axis=1))<=env.radius+1e-7)
                            if previous is not None:
                                # Each next polygon is a subset: area and diameter cannot increase.
                                if metrics['diameter_m']>previous['diameter_m']+1e-5:
                                    raise AssertionError('Diameter increased for nested constraints')
                                if metrics['area_m2']>previous['area_m2']+1e-3:
                                    raise AssertionError('Area increased for nested constraints')
                                transitions.append({'noise':noise,'layout':layout,'seed':env.seed,
                                    'from_n':previous['n'],'to_n':n,
                                    'rounder':metrics['circularity']>previous['circularity']+1e-8,
                                    'less_round':metrics['circularity']<previous['circularity']-1e-8,
                                    'covered_to_failure':not previous['failure'] and metrics['failure'],
                                    'failure_to_covered':previous['failure'] and not metrics['failure']})
                            previous=dict(row)
                            if metrics['failure']:
                                categories=['strongest',f"best_{metrics['vertices_count']}_vertices"]
                                if n==2:
                                    categories.append('two_monitoring_points')
                                if row['polygon_inside_arena']:
                                    categories.append('inside_arena')
                                    if metrics['vertices_count']==4:
                                        categories.append('quadrilateral_inside_arena')
                                if metrics['source_outside']:
                                    categories.append('source_outside')
                                for category in categories:
                                    if category not in examples or metrics['radius_ratio']>examples[category]['metrics']['radius_ratio']:
                                        examples[category]={'seed':env.seed,'noise':noise,'layout':layout,
                                            'n':n,'arena_radius_m':env.radius,'receive_radius_m':env.receive_radius,
                                            'source':env.source.tolist(),'points':points[:n].tolist(),
                                            'measurements':measurements[:n], 'vertices':poly.tolist(),
                                            'polygon_inside_arena':row['polygon_inside_arena'],'metrics':metrics}
                    rows.append(row)
                if (trial+1)%250==0 or trial+1==trials:
                    print(f'{layout}/{noise}: {trial+1}/{trials} scenarios; '
                          f'{len(rows)} intersections; {time.perf_counter()-start:.1f}s',flush=True)
    by_n=aggregate(rows,['layout','noise','n'],confidence=True)
    valid=[r for r in rows if r['status']=='bounded']
    by_edges=aggregate(valid,['layout','noise','vertices_count'])
    by_n_edges=aggregate(valid,['layout','noise','n','vertices_count'])
    transition_groups=defaultdict(list)
    for row in transitions:
        transition_groups[(row['layout'],row['noise'],row['from_n'],row['to_n'])].append(row)
    transition_summary=[]
    for key,group in sorted(transition_groups.items()):
        item=dict(zip(['layout','noise','from_n','to_n'],key))
        item['paired_bounded_scenarios']=len(group)
        for metric in ('rounder','less_round','covered_to_failure','failure_to_covered'):
            item[metric+'_count']=sum(r[metric] for r in group)
        transition_summary.append(item)
    with gzip.open(output/'trials.csv.gz','wt',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=ROW_FIELDS)
        writer.writeheader();writer.writerows(rows)
    write_csv(output/'summary_by_n.csv',by_n)
    write_csv(output/'summary_by_edges.csv',by_edges)
    write_csv(output/'summary_by_n_and_edges.csv',by_n_edges)
    write_csv(output/'nested_transitions.csv',transition_summary)
    (output/'examples.json').write_text(json.dumps(examples,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    metadata={'trials_per_noise_layout':trials,'seed_start':seed,'counts':list(counts),
              'noises':list(noises),'layouts':list(layouts),'intersection_evaluations':len(rows),
              'distinct_source_seeds':trials,'bounded':len(valid),
              'failures':sum(r['failure'] for r in valid),'elapsed_s':time.perf_counter()-start,
              'classification':'max_vertex_distance_to_diameter_midpoint > D/2 + max(1e-8,1e-7*D)',
              'circle_ratio':'2*minimum_enclosing_radius/diameter',
              'circularity':'4*pi*area/perimeter**2',
              'assumptions':['Full bearing polygon, no arena clipping',
                             'One omnidirectional source; receive radius U[1000,1500] m',
                             'Uniform-area source inside radius 1800 m; observations 50 m to receive radius',
                             'Independent bounded fixed errors at exact locations; no spatial correlation',
                             'Same source seeds and geometry paired across noise models',
                             'Counts are nested prefixes, not independent samples within one seed',
                             'Wilson intervals are pointwise across independent seeds within a fixed noise/layout/n',
                             'Pooled edge-count summaries are descriptive and repeat each source across counts',
                             'Synthetic observable layouts; not official simulator output or a search policy'],
              'versions':{'python':platform.python_version(),'numpy':np.__version__,
                          'scipy':version('scipy'),'matplotlib':version('matplotlib')}}
    (output/'metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    return rows,by_n,by_edges,examples,metadata


def plot_example(case,path):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle,Polygon
    p=np.array(case['vertices']);s=np.array(case['source']);points=np.array(case['points'])
    m=case['metrics'];center=np.array(m['diameter_center']);endpoints=np.array(m['diameter_endpoints'])
    mec=np.array(m['mec_center']);radius=m['mec_radius_m']
    fig,axes=plt.subplots(1,2,figsize=(12,5.6),layout='constrained')
    ax=axes[0]
    ax.add_patch(Circle((0,0),1800,fill=False,color='#9ca3af',lw=1))
    colors=['#2563eb','#c47b13','#ad4a77','#65843c','#7555a2']
    for i,(point,reading) in enumerate(zip(points,case['measurements'])):
        angle=np.deg2rad([reading['svd_deg']-1,reading['svd_deg']+1])
        length=max(4000.,float(np.max(np.linalg.norm(p-point,axis=1)))*1.1)
        tips=point+length*np.column_stack((np.cos(angle),np.sin(angle)))
        color=colors[i%len(colors)]
        ax.add_patch(Polygon(np.vstack((point,tips)),fc=color,alpha=.08))
        ax.scatter(*point,s=30,color=color,zorder=5)
        if len(points)<=8:
            ax.annotate(f'S{i+1}',point,xytext=(4,5),textcoords='offset points',fontsize=9)
        for tip in tips:
            ax.plot([point[0],tip[0]],[point[1],tip[1]],lw=.6,color=color,alpha=.7)
    ax.scatter(*s,marker='*',s=150,c='#c47b13',edgecolor='black',lw=.4,zorder=10,label='True source')
    ax.set(xlim=(-2000,2000),ylim=(-2000,2000),title=f"{case['n']} stations | {case['layout']}")
    ax.legend(loc='upper right',fontsize=9)
    ax=axes[1]
    ax.add_patch(Polygon(p,fc='#dbeafe',ec='#2563eb',lw=1.6,label='Feasible polygon'))
    ax.add_patch(Circle(center,m['diameter_radius_m'],fill=False,ec='#c47b13',lw=1.8,ls='--',label='Diameter circle (D/2)'))
    ax.add_patch(Circle(mec,radius,fill=False,ec='#111827',lw=1.5,label='Minimum enclosing circle'))
    ax.plot(endpoints[:,0],endpoints[:,1],':',color='#6b7280',lw=1.2)
    ax.scatter(*center,marker='+',color='#c47b13',s=60)
    ax.scatter(*mec,marker='x',color='#111827',s=40)
    outside=np.array(m['outside_vertex_indices'],dtype=int)
    ax.scatter(p[outside,0],p[outside,1],facecolors='white',edgecolors='#ad4a77',s=70,lw=1.6,zorder=8,label='Uncovered vertex')
    ax.scatter(*s,marker='*',s=130,color='#c47b13',edgecolor='black',lw=.4,zorder=9)
    if len(p)<=8:
        for i,point in enumerate(p):
            close_previous=i>0 and np.min(np.linalg.norm(p[:i]-point,axis=1))<.12*radius
            ax.annotate(f'V{i+1}',point,xytext=(4,-16 if close_previous else 5),
                        textcoords='offset points',fontsize=9)
    half=max(radius*1.5,1.)
    ax.set(xlim=(mec[0]-half,mec[0]+half),ylim=(mec[1]-half,mec[1]+half),
           title=f"{len(p)} vertices | 2R/D = {m['radius_ratio']:.6f}")
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.14),ncol=2,fontsize=8)
    for ax in axes:
        ax.set_aspect('equal');ax.grid(alpha=.15)
        ax.set_xlabel('East x (m)');ax.set_ylabel('North y (m)')
        ax.ticklabel_format(useOffset=False,style='plain')
    fig.suptitle(f"Seed {case['seed']} | {case['noise']} | D={m['diameter_m']:.3f} m, "
                 f"R={radius:.3f} m",fontsize=13)
    fig.savefig(path,dpi=180)
    fig.savefig(path.with_suffix('.svg'))
    plt.close(fig)


def plot_summary(summary,edges,metadata,output):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    styles={'uniform':('#2563eb','o'),'truncnorm':('#c47b13','s')}
    for col,layout in enumerate(metadata['layouts']):
        for noise in metadata['noises']:
            data=sorted((r for r in summary if r['layout']==layout and r['noise']==noise and r['bounded']),key=lambda x:x['n'])
            x=np.array([r['n'] for r in data]);y=np.array([r['failure_rate'] for r in data])
            color,marker=styles[noise]
            axes[0,col].errorbar(x,y,yerr=[y-[r['ci_low'] for r in data],[r['ci_high'] for r in data]-y],
                color=color,marker=marker,capsize=3,lw=1.5,label=noise)
            axes[1,col].plot(x,[r['mean_circularity'] for r in data],color=color,marker=marker,lw=1.5,label=noise)
        axes[0,col].set_title(layout.replace('_',' '))
        for row in (0,1):
            ax=axes[row,col];ax.set_xscale('log',base=2)
            ax.set_xticks(metadata['counts'],[str(n) for n in metadata['counts']])
            ax.set_xlabel('Number of monitoring points');ax.set_ylim(0,1)
            ax.grid(alpha=.18);ax.legend(loc='best',fontsize=9)
        axes[0,col].yaxis.set_major_formatter(PercentFormatter(1))
        axes[0,col].set_ylabel('Cannot cover polygon with any D/2 circle')
        axes[1,col].set_ylabel('Mean circularity: 4 pi A / perimeter²')
    fig.suptitle(f"Synthetic bearing localization | {metadata['trials_per_noise_layout']:,} seeds per panel series\n"
                 'Failure rates: bounded polygons only; bars: pointwise 95% Wilson intervals',fontsize=13)
    fig.savefig(output/'coverage_summary.png',dpi=180)
    fig.savefig(output/'coverage_summary.svg')
    plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for col,layout in enumerate(metadata['layouts']):
        for noise in metadata['noises']:
            data=sorted((r for r in edges if r['layout']==layout and r['noise']==noise
                         and r['bounded']>=100),key=lambda r:r['vertices_count'])
            color,marker=styles[noise]
            axes[col].plot([r['vertices_count'] for r in data],
                           [r['failure_rate'] for r in data],color=color,marker=marker,label=noise)
        axes[col].set(title=layout.replace('_',' '),xlabel='Actual polygon vertex count',
                      ylabel='Diameter-circle failure rate',ylim=(0,1))
        axes[col].set_xticks(range(3,13))
        axes[col].yaxis.set_major_formatter(PercentFormatter(1))
        axes[col].grid(alpha=.18);axes[col].legend(fontsize=9)
    fig.suptitle('Descriptive rates by actual vertex count (groups with at least 100 polygons)\n'
                 'Prefixes pooled: repeated source scenarios; these are not causal effects of vertex count',fontsize=12)
    fig.savefig(output/'failure_by_edges.png',dpi=180)
    fig.savefig(output/'failure_by_edges.svg')
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trials',type=int,default=2500,help='independent source seeds per layout/noise')
    parser.add_argument('--seed',type=int,default=20260911)
    parser.add_argument('--counts',type=int,nargs='+',default=DEFAULT_COUNTS)
    parser.add_argument('--output',type=Path,default=Path(__file__).parent/'output'/'circle_study')
    args=parser.parse_args()
    if args.trials<1 or args.seed<0 or min(args.counts)<2:
        parser.error('trials must be positive, seed nonnegative, counts at least 2')
    counts=tuple(sorted(set(args.counts)))
    rows,summary,edges,examples,metadata=run_study(args.trials,args.seed,counts,
        ('uniform','truncnorm'),('uniform_area','local_walk'),args.output)
    examples['constructed_triangle']=symmetric_counterexample()
    (args.output/'examples.json').write_text(json.dumps(examples,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    plot_summary(summary,edges,metadata,args.output)
    for name in ('strongest','two_monitoring_points','quadrilateral_inside_arena','inside_arena','source_outside','constructed_triangle'):
        if name in examples:
            plot_example(examples[name],args.output/(name+'.png'))
    print(json.dumps(metadata,indent=2),flush=True)


if __name__=='__main__':
    main()

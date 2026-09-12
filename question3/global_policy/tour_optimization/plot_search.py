"""Standalone figures from saved search results; no new simulator calls."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser();p.add_argument('--search',required=True);p.add_argument('--validation',required=True)
    a=p.parse_args();search=Path(a.search);val=Path(a.validation)
    run=json.loads((search/'optimization.json').read_text())
    audit=json.loads((val/'audit.json').read_text())
    plt.rcParams.update({'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False,
                         'font.size':10,'axes.titlesize':13})
    fig,axes=plt.subplots(1,3,figsize=(15,4.7))
    names=[n for n in run['low'] if n!='structural']
    means=[run['low'][n]['delta'] for n in names]
    sem=[run['low'][n]['variance']**.5 for n in names]
    colors=['#21896d' if n.startswith('bo_') else '#607e98' for n in names]
    ax=axes[0]
    for i,(mu,se,color) in enumerate(zip(means,sem,colors)):
        ax.errorbar(i,mu,yerr=se,fmt='o',ms=4,color=color,alpha=.8,capsize=2)
    ax.axhline(0,color='#a1aab0',lw=1);ax.set_title('14-scene paired search (mean +/- SE)')
    ax.set_xlabel('Evaluated configurations; green = GP proposals')
    ax.set_ylabel('Seconds/source vs structural control')
    ax=axes[1]
    for name in run['finalists']:
        values=[run[level][name]['objective'] for level in ('low','medium','high')]
        ax.plot([14,42,84],values,'o-',label=name)
    ax.set_xticks([14,42,84]);ax.set_title('Repeated scenario blocks')
    ax.set_xlabel('Scenes evaluated per configuration');ax.set_ylabel('Mean seconds/source')
    ax.legend(fontsize=8)
    ax=axes[2];names=list(audit['groups'])
    labels={'previous':'Previous tour','structural':'Structural','optimized':'Selected'}
    for i,name in enumerate(names):
        record=audit['groups'][name]['time_s_per_source'];m=record['mean'];ci=record['mean_ci95']
        ax.bar(i,m,color=['#829aaa','#4e9da7','#26866b'][i%3],width=.6)
        ax.errorbar(i,m,yerr=[[m-ci[0]],[ci[1]-m]],fmt='none',ecolor='#283b44',capsize=4)
        ax.text(i,m+11,f'{m:.2f}',ha='center',fontsize=10)
    ax.axhline(200,color='#b25350',ls='--',label='200 target')
    ax.set_xticks(range(len(names)),[labels.get(n,n) for n in names]);ax.set_ylim(0,max(g['time_s_per_source']['mean_ci95'][1] for g in audit['groups'].values())+25)
    ax.set_title('New held-out scenes (95% mean CI)');ax.set_ylabel('Mean seconds/source');ax.legend(fontsize=8)
    fig.suptitle('Q3 local synthetic optimization — development and validation kept separate',fontsize=14)
    fig.tight_layout(rect=(0,0,1,.95))
    out=val/'report';out.mkdir(parents=True,exist_ok=True)
    fig.savefig(out/'optimization.png',dpi=180)
    fig.savefig(out/'optimization.svg')
    print(str(out/'optimization.png'))


if __name__=='__main__':main()

"""Distribution and validation curves; all event probabilities keep tails."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).parent/'results'


def main():
    data=np.load(ROOT/'evaluation_samples.npz')
    info=json.loads((ROOT/'evaluation.json').read_text())
    fig,ax=plt.subplots(1,3,figsize=(16,4.8),layout='constrained')
    for name,color in [('plain','#6c3483'),('orthogonal','#c0392b'),('previous_radius_min','#2874a6')]:
        r=data[name+'_r'].ravel();xs=np.sort(r)
        ax[0].plot(xs,np.arange(1,len(xs)+1)/len(xs),color=color,label=name)
        dev=np.sort(np.abs(r-20));ax[1].plot(dev,np.arange(1,len(dev)+1)/len(dev),color=color,label=name)
        # Use all events in the denominator even when plotting only central range.
        state=np.abs(data[name+'_r']-20).mean((1,2))
        ax[2].plot(np.sort(state),np.arange(1,len(state)+1)/len(state),color=color,label=name)
    ax[0].set(xlim=(0,60),ylim=(0,1),xlabel='MEC radius (m)',ylabel='Cumulative probability',title='Event radius distribution')
    ax[0].axvline(20,color='black',ls='--',lw=1)
    ax[1].set(xlim=(0,30),ylim=(0,1),xlabel='Absolute deviation from 20 m',ylabel='Cumulative probability',title='Target-20 objective distribution')
    ax[2].set(xlim=(0,30),ylim=(0,1),xlabel='State mean absolute deviation (m)',ylabel='Fraction of states',title='State-level performance')
    ax[0].legend(fontsize=8)
    fig.suptitle('Independent test: 2048 states x 256 events, both sides equally weighted; tails retained in probabilities')
    fig.savefig(ROOT/'distributions.png',dpi=180);fig.savefig(ROOT/'distributions.svg');plt.close(fig)
    with (ROOT/'training.csv').open() as f:rows=list(csv.DictReader(f))
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for weight,color in [(0.,'#6c3483'),(.1,'#c0392b'),(.5,'#2874a6')]:
        rr=[r for r in rows if int(r['replicate'])==0 and float(r['orth_weight'])==weight]
        for a,key in zip(axes,['mae20_m','orth']):
            a.plot([int(r['step']) for r in rr],[float(r[key]) for r in rr],color=color,label=f'lambda={weight}')
    axes[0].set(xlabel='Training step',ylabel='Validation MAE from 20 (m)',title='Primary objective')
    axes[1].set(xlabel='Training step',ylabel='Predictive weighted cos squared',title='Orthogonality guidance')
    axes[0].legend();fig.savefig(ROOT/'training_curves.png',dpi=180);plt.close(fig)


if __name__=='__main__':main()

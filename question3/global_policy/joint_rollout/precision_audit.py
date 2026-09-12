"""Reevaluate frozen public states; no new simulator actions or hidden labels."""
import argparse,gzip,json
from dataclasses import replace
from pathlib import Path
import numpy as np
from ..model import World
from .parameters import JointParameters
from .service import choose_action,scenarios
from .kernel import evaluate,exclude_disk


def reconstruct(pack,index,params):
    w=World(params)
    for c in pack['commands'][:index]:
        if c['position'] is None:continue
        q=np.array(c['position']);w.position=q;w.virtual_time=c['response']['virtual_time_s']
        t=w.targets[c['channel']]
        if c['path']=='/measure':
            w.channel=c['channel'];w.coverage.mark(c['channel'],q);t.update_measurement(q,c['response'],params)
        elif c['path']=='/clear':t.update_clear(q,c['response'],params)
    return w


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',default='question3/global_policy/joint_rollout/outputs/development')
    p.add_argument('--count',type=int,default=24);p.add_argument('--late',action='store_true');a=p.parse_args();root=Path(a.input);rows=[]
    files=sorted((root/'combined').glob('*.json.gz'))
    for file in files:
        with gzip.open(file,'rt',encoding='utf8') as f:pack=json.load(f)
        eligible=[e for e in pack['events'] if e['phase']=='local_decision' and e['radius']>20]
        e=(eligible[-1] if a.late else eligible[0]) if eligible else None
        if e is None:continue
        p0=JointParameters(**pack['summary']['parameters']);w=reconstruct(pack,e['command_index'],p0);t=w.targets[e['channel']]
        if t.particles is None:continue
        low=choose_action(w,t)
        w.params=replace(p0,search_scenarios=384,validation_scenarios=2048)
        high=choose_action(w,t)
        sample=scenarios(t,8192,p0.model_seed+pack['summary']['seed']+831007)
        poly=t.polygon.copy()
        for q,r,_ in t.exclusions:poly=exclude_disk(poly,q,r)
        history=np.array([o.position for o in t.observations]).reshape(-1,2)
        results=[]
        for path,q,_,_ in (low,high):
            costs,_=evaluate(poly,w.position,q,path=='/clear',*sample,history,int(w.channel!=t.channel),p0.circle_sides)
            results.append(costs)
        delta=results[0]-results[1]
        row=dict(seed=pack['summary']['seed'],channel=t.channel,command_index=e['command_index'],
            low_action=low[0],high_action=high[0],low_position=low[1].tolist(),high_position=high[1].tolist(),
            action_changed=low[0]!=high[0],position_change_m=float(np.linalg.norm(low[1]-high[1])),
            independent_low_s=float(results[0].mean()),independent_high_s=float(results[1].mean()),
            low_minus_high_s=float(delta.mean()),paired_mc_se=float(delta.std(ddof=1)/np.sqrt(len(delta))))
        rows.append(row);print(json.dumps(row),flush=True)
        if len(rows)>=a.count:break
    result=dict(count=len(rows),states=rows,audit_samples=8192,
        mean_low_minus_high_s=float(np.mean([r['low_minus_high_s'] for r in rows])),
        max_low_minus_high_s=max(r['low_minus_high_s'] for r in rows))
    (root/('precision_audit_late.json' if a.late else 'precision_audit.json')).write_text(json.dumps(result,indent=2),encoding='utf8')


if __name__=='__main__':main()

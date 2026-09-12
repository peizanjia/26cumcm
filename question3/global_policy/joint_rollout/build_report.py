"""Deterministic adverse-case selection and standalone comparison HTML."""
import argparse,gzip,json,subprocess,sys
from pathlib import Path
import numpy as np

LABELS={'baseline':'原版','terminal':'仅分区终点','expectation':'仅期望动作','combined':'两项同时修改'}


def select_cases(rows):
    baseline={r['seed']:r for r in rows['baseline']};combined=rows['combined'];selected={}
    def pick(key,score,count=2,reverse=True):
        for r in sorted(combined,key=lambda r:(score(r),-r['seed']),reverse=reverse)[:count]:
            selected.setdefault(r['seed'],[]).append(key)
    pick('新策略平均耗时最高',lambda r:r['average_time_s'],3)
    pick('相对原版退步最大',lambda r:r['average_time_s']-baseline[r['seed']]['average_time_s'],3)
    pick('长程折返最多',lambda r:r['long_reversals'],2)
    pick('清除miss最多',lambda r:r['misses'],2)
    pick('预测与实际剩余耗时差异较大',lambda r:r['prediction_abs_error_s'] or 0.,2)
    oldworst=sorted(rows['baseline'],key=lambda r:r['average_time_s'],reverse=True)[:2]
    for r in oldworst:selected.setdefault(r['seed'],[]).append('原版耗时最高')
    median=sorted(combined,key=lambda r:r['average_time_s'])[len(combined)//2]
    selected.setdefault(median['seed'],[]).append('中位表现对照')
    return selected


def unpack(root,seed,name,out):
    with gzip.open(root/name/f'{seed}.json.gz','rt',encoding='utf8') as f:pack=json.load(f)
    folder=out/'cases'/str(seed)/name;folder.mkdir(parents=True,exist_ok=True)
    (folder/'summary.json').write_text(json.dumps(pack['summary']),encoding='utf8')
    (folder/'decisions.json').write_text(json.dumps(pack['events']),encoding='utf8')
    (folder/'commands.jsonl').write_text('\n'.join(json.dumps(c) for c in pack['commands']),encoding='utf8')
    (folder/'evaluator').mkdir(exist_ok=True)
    (folder/'evaluator/truth.json').write_text(json.dumps(dict(seed=seed,sources=pack['truth'])),encoding='utf8')
    subprocess.run([sys.executable,'-m','question3.global_policy.replay','--input',str(folder)],check=True,capture_output=True)
    pos=np.zeros(2);segments=[]
    for i,c in enumerate(pack['commands']):
        if c['position'] is None:continue
        q=np.array(c['position'])
        if np.linalg.norm(q-pos)>1e-7:
            reason=c['reason'];kind='clear' if c['path']=='/clear' else ('probe' if reason.startswith(('joint_','diagonal','direct_')) else 'search')
            segments.append(dict(start=pos.tolist(),end=q.tolist(),kind=kind,command=i,
                                 time=c['response']['virtual_time_s'],channel=c['channel'],reason=reason))
        pos=q
    return dict(summary=pack['summary'],segments=segments,truth=pack['truth'],
                sweep=next((e for e in pack['events'] if e['phase']=='sweep_start'),None),
                terminals=[e for e in pack['events'] if e['phase']=='linked_terminal'],
                replay=f'cases/{seed}/{name}/replay.html')


def diagnose(scene):
    old=scene['variants']['baseline'];new=scene['variants']['combined']
    a,b=old['summary'],new['summary'];n=b['true_total'];truth=np.array([[t['x'],t['y']] for t in new['truth']])
    angles=np.sort(np.arctan2(truth[:,1],truth[:,0]));gaps=np.diff(np.r_[angles,angles[0]+2*np.pi])
    distances=np.linalg.norm(truth[:,None]-truth[None,:],axis=2)+np.eye(n)*1e9
    components={}
    for key,f in [('移动',lambda r:r['distance_m']/5),('测量',lambda r:5*r['measures']),
                  ('miss',lambda r:3*r['misses']),('切频',lambda r:r['virtual_time_s']-r['distance_m']/5-5*r['measures']-5*r['true_cleared']-3*r['misses'])]:
        components[key]=(f(b)-f(a))/n
    first=None
    for i,(x,y) in enumerate(zip(old['segments'],new['segments'])):
        if x['channel']!=y['channel'] or np.linalg.norm(np.array(x['end'])-y['end'])>1:
            first=dict(move=i+1,old_channel=x['channel'],new_channel=y['channel'],old_reason=x['reason'],new_reason=y['reason']);break
    return dict(source_count=n,outer_sources=int(np.sum(np.linalg.norm(truth,axis=1)>1500)),
                largest_empty_angle_deg=float(np.degrees(gaps.max())),
                close_pairs_200m=int(np.sum(np.triu(distances<200,1))),
                cost_difference_per_source=components,first_divergence=first)


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',default='question3/global_policy/joint_rollout/outputs/validation')
    p.add_argument('--output',default='question3/global_policy/joint_rollout/outputs/report');a=p.parse_args()
    root=Path(a.input);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    rows=json.loads((root/'cases.json').read_text());summary=json.loads((root/'summary.json').read_text())
    selected=select_cases(rows);scenes={}
    for seed,reasons in selected.items():
        scenes[str(seed)]=dict(reasons=reasons,variants={name:unpack(root,seed,name,out) for name in rows})
        scenes[str(seed)]['diagnosis']=diagnose(scenes[str(seed)])
        print('selected',seed,reasons,flush=True)
    data=dict(labels=LABELS,summary=summary,scenes=scenes,phase='development' if root.name=='development' else 'validation',
        manifest=json.loads((root/'manifest.json').read_text()),
        distribution={name:[dict(seed=r['seed'],value=r['average_time_s'],complete=r['complete']) for r in v] for name,v in rows.items()})
    factorial=root/'factorial_analysis.json'
    if factorial.exists():data['factorial']=json.loads(factorial.read_text(encoding='utf8'))
    (out/'data.json').write_text(json.dumps(data,ensure_ascii=False),encoding='utf8')
    template=Path(__file__).with_name('report_template.html').read_text(encoding='utf8')
    (out/'comparison.html').write_text(template.replace('__DATA__',json.dumps(data,ensure_ascii=False).replace('</','<\\/')),encoding='utf8')
    (out/'case_selection.json').write_text(json.dumps(selected,ensure_ascii=False,indent=2),encoding='utf8')


if __name__=='__main__':main()

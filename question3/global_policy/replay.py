"""Build a local interactive SVG replay of saved public action/observation history."""
import argparse
import base64
import json
from pathlib import Path
import numpy as np
from .model import World,Parameters


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',default='question3/global_policy/outputs/benchmark/seed_20263000')
    args=p.parse_args();folder=Path(args.input)
    summary=json.loads((folder/'summary.json').read_text(encoding='utf8'))
    commands=[json.loads(line) for line in (folder/'commands.jsonl').read_text(encoding='utf8').splitlines()]
    dynamic=summary.get('policy') in ('dynamic_frontier_rollout','route_study','joint_rollout')
    if summary.get('policy')=='joint_rollout':
        from .joint_rollout.parameters import JointParameters
        params=JointParameters(**summary['parameters'])
    elif summary.get('policy')=='route_study':
        from .route_study.parameters import StudyParameters
        params=StudyParameters(**summary['parameters'])
    elif dynamic:
        from .dynamic.parameters import DynamicParameters
        params=DynamicParameters(**summary['parameters'])
    else:params=Parameters(**summary['parameters'])
    world=World(params);frames=[];route=[[0.,0.]];frontiers=[]
    event_file=folder/'decisions.json'
    events=json.loads(event_file.read_text(encoding='utf8')) if event_file.exists() else []
    sweep=next((e for e in events if e['phase']=='sweep_start'),None)
    for i,command in enumerate(commands):
        q=command['position'];channel=command['channel'];r=command['response'];path=command['path']
        if q is not None:
            q=np.asarray(q);world.position=q
            if path=='/measure':
                world.coverage.mark(channel,q);world.targets[channel].update_measurement(q,r,world.params)
            if path=='/clear':world.targets[channel].update_clear(q,r,world.params)
            if np.linalg.norm(q-route[-1])>1e-7:route.append(q.tolist())
            if dynamic and command['reason'] in ('coverage_frontier','inner_search') and not any(np.linalg.norm(q-v)<1e-7 for v in frontiers):frontiers.append(q.tolist())
        state=dict(index=i,time=r['virtual_time_s'],path=path,channel=channel,reason=command['reason'],
                           feedback=r.get('measure_result',r.get('clear_result','')),position=world.position.tolist(),route=list(route),
                           targets=[dict(channel=t.channel,status=t.status if t.status!='unknown' else ('absent' if world.coverage.complete(t.channel) else 'unknown'),
                             polygon=t.polygon.tolist() if t.polygon is not None else None,
                             center=t.center.tolist() if t.center is not None else None,
                             radius=t.radius if np.isfinite(t.radius) else None) for t in world.targets.values()])
        if dynamic:
            state['sweep']=next((e for e in reversed(events) if e['phase'] in ('sweep_start','outer_sweep_start') and e['command_index']<=i+1),sweep)
            state['frontiers']=list(frontiers)
            state['coverage_bits']=[base64.b64encode(np.packbits(row).tobytes()).decode('ascii') for row in world.coverage.covered]
            state['closed_sectors']=[e['sector'] for e in events if e['phase']=='sector_closed' and e['command_index']<=i+1]
            state['decision']=next((e for e in events if e['phase']=='local_decision' and e['command_index']==i),None)
            state['information']=next((e for e in reversed(events) if e['phase']=='stationary_information_value' and e['command_index']<=i),None)
            state['terminal']=next((e for e in reversed(events) if e['phase']=='linked_terminal' and e['command_index']<=i),None)
        frames.append(state)
    truth_file=folder/'evaluator/truth.json'
    truth=json.loads(truth_file.read_text(encoding='utf8'))['sources'] if truth_file.exists() else []
    data=json.dumps(dict(summary=summary,frames=frames,truth=truth,sweep=sweep,
                         coverage_centers=world.coverage.centers.tolist() if dynamic else []),ensure_ascii=False).replace('</','<\\/')
    template=Path(__file__).with_name('replay_template.html').read_text(encoding='utf8')
    (folder/'replay.html').write_text(template.replace('__REPLAY_DATA__',data),encoding='utf8')
    print(f'{len(frames)} frames: {folder / "replay.html"}')


if __name__=='__main__':main()

"""Export public belief states and approximate rollout labels, grouped by scene.

Labels are estimated continuation costs, NOT exact optimal Q or hidden positions.
Raw evaluation truths are deliberately excluded from this export.
"""
import argparse,gzip,json
from dataclasses import asdict
from pathlib import Path
import numpy as np
from ..model import World
from .parameters import JointParameters


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True)
    p.add_argument('--variant',default='expectation');p.add_argument('--limit',type=int,default=1000);a=p.parse_args()
    root=Path(a.input);out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);count=0;seeds=[]
    with gzip.open(out,'wt',encoding='utf8') as output:
        for file in sorted((root/a.variant).glob('*.json.gz'))[:a.limit]:
            with gzip.open(file,'rt',encoding='utf8') as f:pack=json.load(f)
            params=JointParameters(**pack['summary']['parameters']);w=World(params);seed=pack['summary']['seed'];seeds.append(seed)
            decisions={e['command_index']:e for e in pack['events'] if e['phase']=='local_decision'}
            for i,c in enumerate(pack['commands']):
                if i in decisions:
                    e=decisions[i];target=w.targets[e['channel']]
                    labels=[{k:v for k,v in value.items() if k!='search_catalog'} for value in e.get('candidates',[])]
                    state=dict(scene_seed=seed,variant=a.variant,command_index=i,selected_channel=e['channel'],
                        position=w.position.tolist(),measurement_channel=w.channel,sector=e.get('sector'),
                        sweep=next((v for v in pack['events'] if v['phase']=='sweep_start'),None),
                        closed_sectors=[v['sector'] for v in pack['events'] if v['phase']=='sector_closed' and v['command_index']<=i],
                        targets=[t.snapshot() for t in w.targets.values()],
                        unknown_channels=[t.channel for t in w.unknown()],
                        coverage_fraction=w.coverage.summary()['fraction_by_channel'],
                        coverage_cell_size=w.coverage.cell,
                        coverage_packed_hex=[np.packbits(row).tobytes().hex() for row in w.coverage.covered],
                        selected_action=e['action'],selected_position=e['destination'],
                        candidate_labels=labels,label_type='sampled_base_policy_remaining_seconds',
                        split_unit='scene_seed')
                    output.write(json.dumps(state,ensure_ascii=False)+'\n');count+=1
                if c['position'] is None:continue
                q=np.array(c['position']);w.position=q
                if c['path']=='/measure':
                    w.channel=c['channel'];w.coverage.mark(c['channel'],q);w.targets[c['channel']].update_measurement(q,c['response'],params)
                elif c['path']=='/clear':w.targets[c['channel']].update_clear(q,c['response'],params)
    meta=dict(state_count=count,scene_count=len(seeds),scene_seeds=seeds,contains_truth=False,
              schema_version=3,parameters=asdict(params) if seeds else None,
              warning='Approximate teacher labels. Reserve new untouched evaluation seeds after any training on this export.')
    out.with_suffix('.metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf8');print(json.dumps(meta),flush=True)


if __name__=='__main__':main()

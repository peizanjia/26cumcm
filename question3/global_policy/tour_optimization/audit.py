"""Whole-scene statistics, exact source snapshots and cost attribution."""
import argparse,gzip,hashlib,json
from pathlib import Path
import zipfile
import numpy as np
from ..adaptive_mpc.final_audit import group_audit,paired
from ..adaptive_mpc.coverage_audit import inspect_pack,mean_rows


def snapshot_check(path,expected):
    h=hashlib.sha256();count=0;mismatches=[]
    root=Path(__file__).resolve().parents[3]
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            data=archive.read(name);h.update(name.encode());h.update(data);count+=1
            current=root/name
            if not current.exists() or current.read_bytes()!=data:mismatches.append(name)
    return dict(files=count,calculated_hash=h.hexdigest(),expected_hash=expected,
                snapshot_matches_manifest=h.hexdigest()==expected,current_source_differences=mismatches)


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--traces',action='store_true')
    a=p.parse_args();root=Path(a.input)
    cases=json.loads((root/'cases.json').read_text(encoding='utf8'))
    manifest=json.loads((root/'manifest.json').read_text(encoding='utf8'))
    groups={name:group_audit(rows) for name,rows in cases.items()}
    comparisons={}
    names=list(cases)
    for i in range(len(names)):
        for j in range(i):comparisons[f'{names[i]} - {names[j]}']=paired(cases[names[i]],cases[names[j]])
    for name,group in groups.items():
        assert group['seeds']==manifest['seeds'],name
        assert group['maximum_component_accounting_residual_s']<.001,name
        assert len(group['parameter_sets'])==1,name
    result=dict(evaluation=manifest['evaluation'],groups=groups,comparisons=comparisons,
                snapshot=snapshot_check(root/'source_snapshot.zip',manifest['source_hash']),
                manifest=manifest,files_sha256={name:hashlib.sha256((root/name).read_bytes()).hexdigest()
                    for name in ('cases.json','summary.json','manifest.json')})
    assert result['snapshot']['snapshot_matches_manifest']
    if a.traces:
        replay={}
        for name,rows in cases.items():
            records=[]
            for row in rows:
                if not row['complete']:continue
                with gzip.open(root/name/f"{row['seed']}.json.gz",'rt',encoding='utf8') as f:pack=json.load(f)
                record=inspect_pack(pack)
                refine=[e['refinement'] for e in pack['events'] if e['phase']=='global_tour_replan' and 'refinement' in e]
                record['coverage_coordinate_updates']=sum(r['moved'] for r in refine)
                record['forecast_nodes_removed']=sum(r['removed'] for r in refine)
                records.append(record)
            replay[name]=dict(means=mean_rows(records),rows=records)
        result['replay_audit']=replay
    (root/'audit.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    brief={name:dict(mean=g['time_s_per_source']['mean'],ci95=g['time_s_per_source']['mean_ci95'],
                    complete=g['complete'],below200=g['below_200_count'],
                    components={k:v['mean'] for k,v in g['component_s_per_source'].items()}) for name,g in groups.items()}
    print(json.dumps(dict(groups=brief,snapshot=result['snapshot']),indent=2))


if __name__=='__main__':main()

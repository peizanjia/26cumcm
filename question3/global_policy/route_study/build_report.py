"""Build a standalone local comparison, using saved outputs only."""
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).parent
LABELS={'baseline':'原版：插入+2-opt','insertion_2opt':'插入+2-opt','nearest':'最近邻','exact_dp':'固定终点精确DP','exact_open':'自由终点精确DP',
        'inner800':'先内圈800m','inner1100':'先内圈1100m','adaptive_sector':'自适应分区顺序',
        'voi_stops':'原停点VOI','voi_each_stop':'每停点VOI','nearest_voi':'最近邻+每停点VOI',
        'mst_2opt':'MST+2-opt','christofides':'Christofides开环','annealing':'模拟退火',
        'block_first':'块内先扫描后固定清理','global_guided':'全局顺序指导局部','global_graph':'全体已知源图优先清理',
        'endpoint_aware':'无前沿时自由终点'}


def read(path):return json.loads(path.read_text(encoding='utf8'))


def main():
    folders=['screening','combination','open_route','graph_screening','endpoint']
    screening={};locations={}
    for folder in folders:
        result=read(ROOT/'outputs'/folder/'summary.json')
        for name,row in result['variants'].items():
            if name=='baseline' and name in screening:continue
            screening[name]=row;locations[name]=ROOT/'outputs'/folder/name
    locations['baseline']=ROOT.parent/'dynamic/outputs/final_validation'
    scenes={}
    for seed in (20263000,20263001):
        scenes[str(seed)]={}
        for name,folder in locations.items():
            run=folder/f'seed_{seed}'
            commands=[json.loads(line) for line in (run/'commands.jsonl').read_text(encoding='utf8').splitlines()]
            pos=[0.,0.];segments=[];clears=[]
            for c in commands:
                q=c['position']
                if q is None:continue
                if np.linalg.norm(np.array(q)-pos)>1e-7:
                    kind='clear' if c['path']=='/clear' else ('probe' if c['reason'].startswith(('diagonal','direct_measure')) else 'search')
                    segments.append(dict(start=pos,end=q,kind=kind,reason=c['reason']))
                if c['response'].get('clear_result')=='success':clears.append(dict(position=q,channel=c['channel']))
                pos=q
            scenes[str(seed)][name]=dict(segments=segments,clears=clears,summary=read(run/'summary.json'),
                                       truth=read(run/'evaluator/truth.json')['sources'])
    control=read(ROOT.parent/'dynamic/outputs/final_validation/cases.json')['dynamic']
    components=[]
    for r in control:
        n=r['true_total'];movement=r['distance_m']/5;measurement=5*r['measures'];success=5*n;miss=3*r['misses']
        switching=r['virtual_time_s']-movement-measurement-success-miss
        components.append(np.array([movement,measurement,switching,success,miss])/n)
    data=dict(labels=LABELS,screening=screening,holdout=read(ROOT/'outputs/holdout/summary.json')['variants'],
              bounds=read(ROOT/'outputs/bounds.json'),static=read(ROOT/'outputs/static_audit.json'),scenes=scenes,
              components=dict(zip(['movement','measurement','switching','successful_clear','miss'],np.mean(components,axis=0).tolist())))
    out=ROOT/'outputs';(out/'comparison.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf8')
    template=(ROOT/'report_template.html').read_text(encoding='utf8')
    (out/'comparison.html').write_text(template.replace('__DATA__',json.dumps(data,ensure_ascii=False).replace('</','<\\/')),encoding='utf8')
    lines=['|方案|筛选均值s/源|相对原版差值|配对标准误|','|---|---:|---:|---:|']
    for k,r in screening.items():lines.append(f'|{LABELS[k]}|{r["mean_s_per_source"]:.2f}|{r["paired_difference_s"]:+.2f}|{r["paired_se_s"]:.2f}|')
    (out/'screening_table.md').write_text('\n'.join(lines),encoding='utf8')
    print(str(out/'comparison.html'))


if __name__=='__main__':main()

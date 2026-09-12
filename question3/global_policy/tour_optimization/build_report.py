"""Compact standalone trace report; omit redundant optimizer search catalogs."""
import argparse,gzip,json
from pathlib import Path
from ..adaptive_mpc.build_report import compact_pack,cost_breakdown,select_cases


LABELS={'previous':'上轮冻结 tour','structural':'覆盖坐标优化＋开放路线',
        'pilot':'首批参数候选','optimized':'本轮最终候选','refine_exact':'覆盖坐标优化＋开放路线'}


def strip_catalogs(value):
    if isinstance(value,list):return [strip_catalogs(v) for v in value]
    if isinstance(value,dict):return {k:strip_catalogs(v) for k,v in value.items() if k!='search_catalog'}
    return value


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--case-count',type=int,default=3)
    a=p.parse_args();root=Path(a.input);out=root/'report';out.mkdir(parents=True,exist_ok=True)
    cases=json.loads((root/'cases.json').read_text(encoding='utf8'))
    summary=json.loads((root/'summary.json').read_text(encoding='utf8'))
    manifest=json.loads((root/'manifest.json').read_text(encoding='utf8'))
    selected=select_cases(cases,a.case_count);scenes={}
    for seed,reasons in selected.items():
        variants={}
        for name in cases:
            with gzip.open(root/name/f'{seed}.json.gz','rt',encoding='utf8') as f:pack=json.load(f)
            variants[name]=strip_catalogs(compact_pack(pack))
            variants[name]['cost_s_per_source']=cost_breakdown(pack['summary'])
        scenes[str(seed)]=dict(reasons=reasons,variants=variants)
    data=dict(labels={n:LABELS.get(n,n) for n in cases},names=list(cases),rows=cases,summary=summary,
              manifest=manifest,scenes=scenes,case_order=list(scenes),missing_traces=[],reference=None,
              input=str(root),evaluation=manifest['evaluation'])
    template=(Path(__file__).parent.parent/'adaptive_mpc/report_template.html').read_text(encoding='utf8')
    template=template.replace('对照为旧 combined 策略；比较全局动作估值、扇区路线与清除/覆盖共路三种实现。',
                              '对照为上轮冻结 tour；本轮依次比较覆盖停点坐标优化与昂贵多场景参数优化。')
    template=template.replace('逐步重规划：停下来之后，重新决定下一步','继续优化：覆盖停点与多场景参数搜索')
    template=template.replace('fmt(r.expected_remaining_s),fmt(r.service_rollout_s',
                              'fmt(r.selection_score_s??r.expected_remaining_s),fmt(r.service_rollout_s')
    template=template.replace('本步表中估值只含选中源的局部期望续策；全局选源由上方开放路线决定。这不是全场剩余时间。',
        '动作评分＝本源期望＋回访软罚－耦合权重×旁扫净收益；旁扫列列出乘权重前的秒数。全局选源仍由开放路线决定，不代表全场剩余时间。')
    (out/'comparison.html').write_text(template.replace('__REPORT_DATA__',json.dumps(data,ensure_ascii=False).replace('</','<\\/')),encoding='utf8')
    (out/'case_selection.json').write_text(json.dumps(selected,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(dict(report=str(out/'comparison.html'),cases=len(scenes),bytes=(out/'comparison.html').stat().st_size)))


if __name__=='__main__':main()

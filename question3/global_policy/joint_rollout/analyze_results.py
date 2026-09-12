"""Factorial paired effects and scene-stratified diagnostics after frozen evaluation."""
import argparse,json,math
from pathlib import Path
import numpy as np
from scipy.stats import t,ttest_1samp


def paired(d):
    d=np.asarray(d);n=len(d);se=float(np.std(d,ddof=1)/math.sqrt(n));half=float(t.ppf(.975,n-1)*se)
    return dict(count=n,mean=float(d.mean()),se=se,ci95=[float(d.mean()-half),float(d.mean()+half)],
                faster=int(np.sum(d<-1e-6)),same=int(np.sum(np.abs(d)<=1e-6)),slower=int(np.sum(d>1e-6)),
                p_value=float(ttest_1samp(d,0).pvalue) if se else 1.,
                std=float(d.std(ddof=1)),p05=float(np.quantile(d,.05)),p95=float(np.quantile(d,.95)))


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',default='question3/global_policy/joint_rollout/outputs/validation');a=p.parse_args();root=Path(a.input)
    rows=json.loads((root/'cases.json').read_text());ids=sorted(r['seed'] for r in rows['baseline']);lookup={k:{r['seed']:r for r in v} for k,v in rows.items()}
    if not all(lookup[k][s]['complete'] for k in lookup for s in ids):raise RuntimeError('Incomplete episodes: analyze completion and timing separately before comparing')
    v={k:np.array([lookup[k][s]['average_time_s'] for s in ids]) for k in lookup}
    definitions={
        'terminal_without_expectation':('原动作下：改分区 vs 不改',v['terminal']-v['baseline']),
        'terminal_with_expectation':('期望动作下：改分区 vs 不改',v['combined']-v['expectation']),
        'expectation_without_terminal':('原分区下：改动作 vs 不改',v['expectation']-v['baseline']),
        'expectation_with_terminal':('新分区下：改动作 vs 不改',v['combined']-v['terminal']),
        'interaction':('两项交互作用',v['combined']-v['expectation']-v['terminal']+v['baseline'])}
    effects={name:{'label':label,**paired(d)} for name,(label,d) in definitions.items()}
    previous=0.
    for i,(pv,name) in enumerate(sorted((r['p_value'],name) for name,r in effects.items())):
        previous=max(previous,min(1.,pv*(len(effects)-i)));effects[name]['holm_p']=previous
    strata={}
    for n in range(10,17):
        mask=np.array([lookup['baseline'][s]['true_total']==n for s in ids]);strata[str(n)]={}
        for k in ('terminal','expectation','combined'):
            strata[str(n)][k]=paired((v[k]-v['baseline'])[mask])
    result=dict(factorial_effects=effects,by_source_count=strata,statistical_unit='scene',
        all_complete=True,scenario_count=len(ids),note='95% intervals are marginal; Holm p-values adjust the five factorial contrasts.')
    (root/'factorial_analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    lines=['# 冻结配置配对结果','',f'场景数：{len(ids)}；全部为本地合成。负差值表示改动更快。','',
           '| 对比 | 差值 秒/源 | 95%配对区间 | Holm p | 更快/相同/更慢 |','|---|---:|---|---:|---|']
    for r in effects.values():lines.append(f"| {r['label']} | {r['mean']:.4f} | [{r['ci95'][0]:.4f}, {r['ci95'][1]:.4f}] | {r['holm_p']:.6g} | {r['faster']}/{r['same']}/{r['slower']} |")
    lines+=['','区间反映当前生成分布下的不确定性，不是官方成绩或逐场改善保证。训练使用这些种子后应另留最终测试集。']
    (root/'factorial_results.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print(json.dumps(effects,ensure_ascii=False,indent=2))


if __name__=='__main__':main()

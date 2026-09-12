"""Expand every nonzero first-observation branch into a next-action table."""
import argparse
import csv
import html
import json
from pathlib import Path
import numpy as np
from .solve_single import Belief, outcomes, likelihood, candidate_actions, action_value


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',default='question3/local_sim/outputs/single')
    args=parser.parse_args()
    folder=Path(args.input)
    data=json.loads((folder/'tree.json').read_text(encoding='utf8'))
    cfg=data['configuration']; first=data['best']; q=np.array(first['position'])
    b=Belief(cfg['particles'],cfg['seed']); p,w=b.planning(cfg['planning'])
    obs=list(outcomes(p,q,cfg['bin_width'])) if first['kind']=='measure' else [
        ('success',dict(clear_result='success'),None),('no_target_in_range',dict(clear_result='no_target_in_range'),None)]
    rows=[]; expanded=[]
    for label,response,width in obs:
        weights=w*likelihood(p,q,response,width or .01)
        prob=float(weights.sum())
        if prob<=1e-12: continue
        keep=weights>0; pp=p[keep]; ww=weights[keep]/prob
        if label=='success':
            best=dict(kind='stop',position=q.tolist(),expected_s=0,branches=[])
        elif label=='near':
            best=dict(kind='clear',position=q.tolist(),expected_s=5,branches=[dict(observation='success',probability=1,continuation_s=0)])
        else:
            values=[action_value(a,pp,ww,q,cfg['bin_width'],1,(np.zeros(2),q))
                    for a in candidate_actions(pp,ww,q,(np.zeros(2),q))]
            best=min(values,key=lambda r:r['expected_s'])
        expanded.append(dict(observation=label,probability=prob,next_action=best))
        rows.append(dict(observation=label,probability=prob,action=best['kind'],
                         x=best['position'][0],y=best['position'][1],remaining_estimate_s=best['expected_s']))
    (folder/'expanded_tree.json').write_text(json.dumps(expanded,indent=2),encoding='utf8')
    with (folder/'branches.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    cards=[]
    for row,branch in zip(rows,expanded):
        body=''.join(f"<tr><td>{html.escape(c['observation'])}</td><td>{c['probability']:.2%}</td><td>{c['continuation_s']:.2f} 秒</td></tr>" for c in branch['next_action']['branches'])
        cards.append(f"<details><summary>{html.escape(row['observation'])} · 概率 {row['probability']:.2%} → {row['action']} ({row['x']:.1f}, {row['y']:.1f})</summary><p>该分支重新规划的剩余时间估计：{row['remaining_estimate_s']:.2f} 秒</p><table><tr><th>下一次反馈</th><th>条件概率</th><th>叶节点清扫估计</th></tr>{body}</table></details>")
    summary=json.loads((folder/'summary.json').read_text(encoding='utf8'))
    stats=''.join(f"<tr><td>{k}</td><td>{summary[k]['mean_s']:.2f}</td><td>{summary[k]['successes']}/{summary[k]['n']}</td></tr>" for k in ('optimized_first','straight_first','conservative'))
    document=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>单目标探测与清除决策树</title>
<style>body{{font:16px/1.65 system-ui,sans-serif;max-width:1000px;margin:36px auto;padding:0 20px;color:#193348;background:#f5f8fb}}h1{{font-size:28px}}details{{background:white;border:1px solid #cad8e3;border-radius:8px;padding:12px 18px;margin:10px 0}}summary{{cursor:pointer;font-weight:600}}table{{border-collapse:collapse;width:100%;background:white}}td,th{{padding:6px 12px;border-bottom:1px solid #dce5ed;text-align:left}}.root{{padding:18px;background:#163a54;color:white;border-radius:10px}}p{{max-width:900px}}</style>
<h1>先斜前探测，再按反馈决定</h1><p>本地合成实验。原点已获得0°示向度，随机隐藏半径1000–1500m，扇形内目标面积均匀。时间从初次示向度已知后开始。</p>
<div class="root">第一动作：{first['kind']} ({q[0]:.2f}, {q[1]:.2f}) m<br>有限粒子树估计：{first['expected_s']:.2f}秒（不是全局最优值）</div>
<p>点击任一分支展开下一次反馈。下表采用4°观测分箱；实际策略使用0.01°读数重新规划，因此不会机械照搬粗分箱动作。未出现的分支仅表示本次粒子估计概率为零。near直接原地清除；no_signal更新接收半径后验，不认定目标消失。</p>
<table><tr><th>配对策略</th><th>平均剩余时间/秒</th><th>清除成功</th></tr>{stats}</table><h2>所有非零概率分支</h2>{''.join(cards)}
<p>叶节点为覆盖所有剩余粒子的两种方向清扫中较优者，只对离散粒子模型可行。连续目标若遇粒子耗尽或20轮仍未成功，执行覆盖整个初始扇形的25m网格清除。多源全局探索、官方演练和连续空间全局最优认证均不包含在本结果中。</p></html>'''
    (folder/'decision_tree.html').write_text(document,encoding='utf8')
    print(f'Exported {len(rows)} branches; probability sum={sum(r["probability"] for r in rows):.12f}')


if __name__=='__main__':main()

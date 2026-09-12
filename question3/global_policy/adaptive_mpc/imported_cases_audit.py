"""Reproduce the14 imported diagnostic-case comparisons from public logs only."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np


def analyze(path):
    with gzip.open(path,'rt',encoding='utf8') as stream:
        pack=json.load(stream)
    # Do not use evaluator truth or recorded summary scores. Recompute all
    # reported costs and source counts from accepted public commands/events.
    commands=pack['commands'];events=pack['events']
    previous=np.zeros(2);channel=1;discovered={};cleared=set();moves=[];stops=[]
    components=dict(movement=0.,measure=0.,switch=0.,clear=0.)
    unknown_scans=known_scans=measures=misses=0
    final_time=last_clear=0.;last_clear_command=None
    for index,command in enumerate(commands):
        response=command['response']
        if not response.get('accepted'):continue
        final_time=float(response['virtual_time_s'])
        if command['position'] is None:continue
        point=np.asarray(command['position']);target=command['channel']
        distance=float(np.linalg.norm(point-previous))
        components['movement']+=distance/5
        if not stops or distance>1e-7:
            stops.append(dict(first_command=index+1,last_command=index+1,position=point.tolist(),
                              movement_m=distance,known_scans=0,unknown_scans=0,new_channels=[],reasons=[]))
        stop=stops[-1];stop['last_command']=index+1;stop['reasons'].append(command['reason'])
        if distance>1e-7:
            moves.append(dict(command=index+1,start=previous.tolist(),end=point.tolist(),distance_m=distance,
                              time_s=final_time,channel=target,reason=command['reason'],path=command['path']))
        if command['path']=='/measure':
            measures+=1;components['measure']+=5;components['switch']+=int(channel!=target);channel=target
            if target in discovered:known_scans+=1;stop['known_scans']+=1
            else:unknown_scans+=1;stop['unknown_scans']+=1
            if response['measure_result'] in ('direction','near') and target not in discovered:
                discovered[target]=dict(command=index+1,time_s=final_time,position=point.tolist())
                stop['new_channels'].append(target)
        elif command['path']=='/clear':
            success=response['clear_result']=='success'
            components['clear']+=5 if success else 3
            if success:
                cleared.add(target);last_clear=final_time;last_clear_command=index+1
            else:misses+=1
        previous=point
    if not cleared or not any(event['phase']=='coverage_closed' for event in events):
        raise ValueError(f'Incomplete imported case: {path}')
    if abs(sum(components.values())-final_time)>1e-3:
        raise ArithmeticError(f'Public command cost decomposition mismatch: {path}')
    for stop in stops:stop['reasons']=dict(Counter(stop['reasons']))
    return dict(seed=int(path.name.split('.')[0]),variant=path.parent.name,input=str(path),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),source_count=len(cleared),cleared_channels=sorted(cleared),
                virtual_time_s=final_time,average_s=final_time/len(cleared),components_s=components,
                distance_m=components['movement']*5,measures=measures,unknown_scans=unknown_scans,known_scans=known_scans,
                scan_and_switch_s=components['measure']+components['switch'],misses=misses,
                final_clear_time_s=last_clear,final_clear_command=last_clear_command,
                post_clear_coverage_s=final_time-last_clear,
                post_clear_movement_s=sum(move['distance_m']/5 for move in moves if move['command']>last_clear_command),
                longest_leg=max(moves,key=lambda move:move['distance_m']),stops=stops,discoveries=discovered,
                phase_counts=dict(Counter(event['phase'] for event in events)),
                tour_replans=[event for event in events if event['phase']=='global_tour_replan'])


def xy(point):return f'({point[0]:.1f}, {point[1]:.1f})'


def render(report):
    rows=report['pairs'];old=report['aggregates']['combined'];new=report['aggregates']['tour_candidate']
    text=['# 导入的14个历史差案例：逐场复核','',
          '这批种子来自用户提供的 `comparison.html`，是有意选择的差案例；它检验既有问题是否得到改善，不能代替独立随机验证，也不能据此宣布总体平均低于200秒。下表“旧”是本环境重跑的 `combined`，“新”是冻结候选 `tour_candidate`，不是直接照抄旧HTML成绩。全部为本地合成运行。','',
          f'14场两组都完成了175个源的清除；新方案13场更快、1场更慢。这批案例的逐场平均为 {old["average_s"]:.2f} → {new["average_s"]:.2f} 秒/源（{new["average_s"]-old["average_s"]:+.2f}），仅1场新成绩低于200秒/源。','',
          '数字由接受成功的 `commands` 和 `events` 重新计算，统计不使用隐藏源坐标或summary中的成绩。命令编号从1开始，包含`/enter`；坐标单位为米。移动时间按距离÷5；扫频成本包含每次测量5秒及实际切频1秒。“尾段”指最后一次成功清除之后直到覆盖完成的时间，不包括最后一次清除命令自身。尾段仍可能是必要的不存在性证明，不能直接当作全部可删除的浪费。','',
          '| 种子 | 源数 | 旧→新（秒/源） | 差值（新−旧） | 移动距离变化（米/场） | 扫描次数变化 | 扫频含切频变化（秒/场） | 覆盖尾段旧→新（秒/场） |',
          '|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        a=row['combined'];b=row['tour_candidate'];d=row['delta']
        text.append(f'| {row["seed"]} | {a["source_count"]} | {a["average_s"]:.2f} → {b["average_s"]:.2f} | {d["average_s"]:+.2f} | {d["distance_m"]:+.1f} | {d["measures"]:+d} | {d["scan_and_switch_s"]:+.0f} | {a["post_clear_coverage_s"]:.1f} → {b["post_clear_coverage_s"]:.1f} |')
    text += ['', '## 每场判断', '', '各场附一段可定位的旧轨迹，并列出新方案最大单腿。单腿长度仅用于核查，不代表它独自造成全部时间差；最长单腿增加也可能与全程缩短同时发生。', '']
    for row in rows:
        a=row['combined'];b=row['tour_candidate'];d=row['delta'];leg=a['longest_leg']
        if d['average_s']>0:
            explanation=f'唯一退步例。多扫{d["measures"]}次，扫频含切频多付{d["scan_and_switch_s"]:.0f}秒，移动另增{d["distance_m"]/5:.1f}秒；详细证据见下节。'
        else:
            explanation=f'主要收益来自移动减少{-d["distance_m"]:.1f}米，折合{-d["distance_m"]/5:.1f}秒。'
            if d['scan_and_switch_s']>0:explanation+=f'扫频额外花费{d["scan_and_switch_s"]:.0f}秒，抵消部分收益。'
            elif d['scan_and_switch_s']<0:explanation+=f'扫频再节省{-d["scan_and_switch_s"]:.0f}秒。'
            if d['post_clear_coverage_s']>1:
                explanation+=f'尾段反而增加{d["post_clear_coverage_s"]:.1f}秒，说明本场总收益不等于尾段改善。'
            elif a['post_clear_coverage_s']>1:
                explanation+=f'尾段减少{-d["post_clear_coverage_s"]:.1f}秒。'
            else:explanation+='两组尾段均为0，收益来自清除完成之前。'
        text.append(f'- **{row["seed"]}**：{explanation}轨迹定位：旧第{leg["command"]}条由{xy(leg["start"])}到{xy(leg["end"])}，单腿{leg["distance_m"]:.1f}米；新最大单腿{b["longest_leg"]["distance_m"]:.1f}米。')
    regression=next(row for row in rows if row['delta']['average_s']>0)
    a=regression['combined'];b=regression['tour_candidate'];d=regression['delta'];n=a['source_count']
    text += ['', '## 唯一退步例20271390的命令证据', '',
             f'本场10个源，{a["average_s"]:.2f} → {b["average_s"]:.2f}秒/源，总时间增加{b["virtual_time_s"]-a["virtual_time_s"]:.3f}秒。成本账严格分解为：移动+{d["distance_m"]/5:.3f}秒，测量+{d["measures"]*5:.0f}秒，切频+{b["components_s"]["switch"]-a["components_s"]["switch"]:.0f}秒，多一次清除失败+3秒。测量和切频合计+209秒，占新增时间82.6%；这是日志成本归因，不等同于已经证明某个参数造成82.6%的退步。','',
             '真实轨迹显示的机制是未知频道扫频批次增多，同时探索先后次序发生了变化：','',
             f'- 已知频道扫描 {a["known_scans"]} → {b["known_scans"]} 次；尚未发现的频道扫描 {a["unknown_scans"]} → {b["unknown_scans"]} 次。新增开销主要来自未知频道，不是正交补测已知频道过多。',
             '- 新第22–41条在 **(-553.8, -26.6)** 建图，其中14次面向未知频道，没有发现新源。随后第63–78条在 **(-840.0, 360.0)**、第79–92条在 **(-600.0, 840.0)** 各扫描14个未知频道，也都返回无信号。这些测量有覆盖证明价值，不能因未发现源就判为无效；但它们确实付出了批量扫频成本。',
             '- 新`global_tour_replan`在`command_index=62`明确选择 **(-840, 360)**，在`command_index=78`选择 **(-600, 840)**，记录的动作类型均为`coverage`。旧第47–60条则在 **(-840, -360)** 扫频，之后向南推进。日志支持“本场向北/向南的探索顺序不同”，没有给出另一个顺序在相同观测历史下的反事实实验。',
             f'- 对同一频道7：旧第{a["discoveries"][7]["command"]}条在{xy(a["discoveries"][7]["position"])}、t={a["discoveries"][7]["time_s"]:.3f}秒首次发现；新第{b["discoveries"][7]["command"]}条在{xy(b["discoveries"][7]["position"])}、t={b["discoveries"][7]["time_s"]:.3f}秒才发现。另一方面，频道12由旧t={a["discoveries"][12]["time_s"]:.3f}提前到新t={b["discoveries"][12]["time_s"]:.3f}秒发现。因此不能概括成新方案让所有源都发现更晚。',
             f'- 新的尾段实际上更短：{a["post_clear_coverage_s"]:.3f} → {b["post_clear_coverage_s"]:.3f}秒。旧最后清除在第{a["final_clear_command"]}条、t={a["final_clear_time_s"]:.3f}；新在第{b["final_clear_command"]}条、t={b["final_clear_time_s"]:.3f}。本场退步发生在最终清除之前，不能套用开发v2“主要慢在收尾”的结论。','',
             '可据此提出后续假设：全局路线是否过早支付了多批未知频道扫描费用，以及是否应比较两种周向探索顺序。单次配对轨迹无法区分路线先后、局部观测误差、候选评估和扫频阈值各自的因果贡献；需要固定其他模块的配对消融验证。本轮冻结策略，仅审计记录，没有据此调整代码。','',
             '## 数据与复现','',
             '- 原始28份压缩记录：[imported_cases](outputs/imported_cases/)。唯一退步例：[旧记录](outputs/imported_cases/combined/20271390.json.gz)、[新记录](outputs/imported_cases/tour_candidate/20271390.json.gz)。',
             '- 逐场统计、命令坐标、首次发现时间与输入SHA256：[imported_cases_audit.json](outputs/imported_cases_audit.json)。',
             '- 分析脚本：[imported_cases_audit.py](imported_cases_audit.py)。没有修改策略与冻结源代码快照。','',
             '```powershell',
             r'.\.venv\Scripts\python.exe -m question3.global_policy.adaptive_mpc.imported_cases_audit',
             '```','']
    return '\n'.join(text)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',default='question3/global_policy/adaptive_mpc/outputs/imported_cases')
    parser.add_argument('--json',default='question3/global_policy/adaptive_mpc/outputs/imported_cases_audit.json')
    parser.add_argument('--markdown',default='question3/global_policy/adaptive_mpc/IMPORTED_CASES.md')
    args=parser.parse_args();root=Path(args.input)
    variants={variant:{int(path.name.split('.')[0]):analyze(path) for path in (root/variant).glob('*.json.gz')}
              for variant in ('combined','tour_candidate')}
    if set(variants['combined'])!=set(variants['tour_candidate']):raise ValueError('Unpaired imported seeds')
    pairs=[]
    for seed in sorted(variants['combined']):
        old=variants['combined'][seed];new=variants['tour_candidate'][seed]
        if old['cleared_channels']!=new['cleared_channels']:raise ValueError('Source-clear sets differ')
        keys=('average_s','distance_m','measures','scan_and_switch_s','post_clear_coverage_s')
        pairs.append(dict(seed=seed,combined=old,tour_candidate=new,delta={key:new[key]-old[key] for key in keys}))
    aggregates={variant:{key:float(np.mean([row[key] for row in values.values()]))
                         for key in ('average_s','distance_m','measures','scan_and_switch_s','post_clear_coverage_s')}
                for variant,values in variants.items()}
    report=dict(evaluation='selected_historical_bad_cases_local_synthetic',count=len(pairs),
                source_total=sum(row['combined']['source_count'] for row in pairs),
                command_numbering='one-based commands array, includes enter',costs='movement distance/5 + measure5 + actual switch1 + clear success5/failure3',
                aggregates=aggregates,pairs=pairs)
    Path(args.json).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    Path(args.markdown).write_text(render(report),encoding='utf8')
    print(json.dumps(dict(count=len(pairs),source_total=report['source_total'],aggregates=aggregates,
                         faster=sum(row['delta']['average_s']<0 for row in pairs),
                         regression_seeds=[row['seed'] for row in pairs if row['delta']['average_s']>0]),indent=2))


if __name__=='__main__':main()

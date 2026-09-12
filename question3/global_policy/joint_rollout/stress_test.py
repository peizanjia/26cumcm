"""Constructed geometry diagnostics, deliberately separate from IID statistics."""
from .benchmark import metrics
import argparse,gzip,json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
import numpy as np
from question3.local_sim.simulator import Simulator,Client,Source
from .planner import Planner
from .parameters import JointParameters
from .run import VARIANTS

DESCRIPTIONS={
    'boundary':'12个源沿1730m边界分布，接收半径1000–1120m，原点全部不可见。',
    'three_clusters':'12个源集中于三个相距较远的小簇，检验簇内清理与跨簇衔接。',
    'one_sided':'10个源全部位于同一侧狭长区域，仍需认证其余未知频道，检验空区搜索成本。',
    'mixed_scales':'中心附近、内圈和边界各4个源，接收半径交替1000/1500m，检验不同尺度信息。',
}


def sources_for(name):
    rng=np.random.default_rng(4041);points=[]
    if name=='boundary':
        for i in range(12):
            a=2*np.pi*i/12+.13;points.append((1730*np.cos(a),1730*np.sin(a),1000+10*i))
    elif name=='three_clusters':
        for center in ((1200,0),(-1000,650),(0,-1300)):
            for _ in range(4):
                x,y=np.array(center)+rng.uniform(-80,80,2);points.append((x,y,rng.uniform(1000,1500)))
    elif name=='one_sided':
        for i in range(10):points.append((1100+65*i,(-1)**i*(40+12*i),1000+15*i))
    elif name=='mixed_scales':
        for i in range(12):
            r=(120,900,1710)[i//4];a=2*np.pi*(i%4)/4+.22*(i//4)
            points.append((r*np.cos(a),r*np.sin(a),1000 if i%2 else 1500))
    return [Source(1+i,*map(float,p)) for i,p in enumerate(points)]


def task(args):
    name,variant,folder=args;sim=Simulator(20279900+list(DESCRIPTIONS).index(name),sources=sources_for(name))
    planner=Planner(Client(simulator=sim),JointParameters(**VARIANTS[variant]));row=planner.run()
    row.update(seed=sim.seed,variant=variant,scenario_family=name,evaluation='constructed_stress_not_iid',
               true_total=len(sim.sources),true_cleared=sum(s.cleared for s in sim.sources))
    if row['complete'] and row['true_total']!=row['true_cleared']:row.update(complete=False,failure='Evaluator detected incomplete clearing')
    row=metrics(planner,row);out=Path(folder)/name/variant;planner.save(out,row);sim.dump(out/'evaluator')
    return row


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',default='question3/global_policy/joint_rollout/outputs/stress');p.add_argument('--report-only',action='store_true');a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    args=[(name,variant,str(out)) for name in DESCRIPTIONS for variant in VARIANTS]
    if a.report_only:rows=json.loads((out/'cases.json').read_text())
    else:
        with ProcessPoolExecutor(max_workers=2) as pool:rows=list(pool.map(task,args))
        (out/'cases.json').write_text(json.dumps(rows,indent=2),encoding='utf8')
    import subprocess,sys,html
    blocks=[]
    for name,description in DESCRIPTIONS.items():
        lines=[]
        for r in rows:
            if r['scenario_family']!=name:continue
            folder=out/name/r['variant']
            if not (folder/'replay.html').exists():subprocess.run([sys.executable,'-m','question3.global_policy.replay','--input',str(folder)],check=True,capture_output=True)
            lines.append(f"<tr><td><a href='{name}/{r['variant']}/replay.html'>{r['variant']}：完整回放</a></td><td>{r['true_cleared']}/{r['true_total']}</td><td>{r['average_time_s']:.2f}</td><td>{r['distance_m']/1000:.2f}</td><td>{r['measures']}/{r['misses']}</td><td>{r['long_reversals']}</td></tr>")
        shapes=['<circle cx="0" cy="0" r="1800" fill="#f5f8fb" stroke="#9cb1c0" stroke-width="7"/>']
        for variant,color,width in [('baseline','#b5c2cc',17),('combined','#23789d',9)]:
            points=[[0.,0.]]
            for text in (out/name/variant/'commands.jsonl').read_text().splitlines():
                c=json.loads(text)
                if c['position'] is not None:points.append(c['position'])
            coordinates=' '.join(f'{x},{-y}' for x,y in points)
            shapes.append(f'<polyline points="{coordinates}" fill="none" stroke="{color}" stroke-width="{width}" stroke-opacity=".75"/>')
        for t in sources_for(name):
            shapes.append(f'<circle cx="{t.x}" cy="{-t.y}" r="24" fill="#b33666"/><text x="{t.x+25}" y="{-t.y-25}" font-size="65" fill="#aa3461">{t.channel}</text>')
        svg='<svg viewBox="-2000 -2000 4000 4000" style="display:block;width:100%;max-width:620px;margin:auto">'+''.join(shapes)+'</svg><p>灰线：原版；蓝线：两项同时修改；红点：离线真实源。</p>'
        blocks.append(f'<section><h2>{html.escape(name)}</h2><p>{description}</p>'+svg+'<table><tr><th>方案</th><th>清除</th><th>秒/源</th><th>路程km</th><th>测量/miss</th><th>长折返</th></tr>'+''.join(lines)+'</table></section>')
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>显著几何特征压力案例</title><style>body{max-width:1100px;margin:30px auto;font:16px/1.6 system-ui;background:#eef3f7;color:#183448;padding:20px}section{background:white;border-radius:12px;padding:20px;margin:18px 0}table{border-collapse:collapse;width:100%}td,th{padding:8px;border-bottom:1px solid #dfe8ef;text-align:left}a{color:#176eae}</style><h1>显著几何特征压力案例</h1><p>人工构造的诊断场景，全部满足源数、频道唯一和物理半径范围。这些位置不按原随机分布抽样，不能混入1000场统计，也不用于声称总体平均改善。点击各方案查看凸多边形、路径与期望动作。</p>'''+''.join(blocks)+'</html>'
    (out/'index.html').write_text(page,encoding='utf8')
    print(json.dumps([{k:r[k] for k in ('scenario_family','variant','complete','average_time_s','failure')} for r in rows],indent=2),flush=True)


if __name__=='__main__':main()

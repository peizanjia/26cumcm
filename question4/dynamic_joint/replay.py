"""Extend the verified Q3-style replay with dynamic map and shared-stop values."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from question4.full_mission.replay import render as render_full


EXTENSION = r"""
// Dynamic data is recorded policy state. Hidden evaluator truth never enters
// this layer's candidate or unknown-map calculations.
Object.assign(REASONS,{dynamic_direction_frontier:'未知方向缺口候选',dynamic_coverage_completion:'连续覆盖补全测点',
 multiple_source_shared_point:'多源共享测点',valuable_intermediate_stop:'途中高价值停测',current_stop_reassessment:'当前停点重新评估',
 multisource_refined_point:'联合价值调整测点',shared_stop_certified_clear:'同停点有保证清除',reassessed_known_value:'到点后重评已知源补测',
 required_continuous_coverage_scan:'连续覆盖所需扫描',reassessed_unknown_value:'到点后重评未知频道',
 forward_lateral:'斜前方交会测量',center_lateral:'定位中心侧向测量',short_probe:'短距离试探',
 posterior_center:'后验中心测量',mec_center:'包围圆中心测量',positive_anchor_probe:'从既有接收点短程探测',
 onward_service_probe:'兼顾后续行程的定位测量'});
const actionLabel=k=>({measure:'测量',clear:'清除'}[k]??k);
function dynamicCoverage(f) {return f?.coverage_snapshot??f?.coverage??(f?.unknown_domains?{channels:Object.fromEntries(f.unknown_domains.map(c=>[String(c.channel),c]))}:null)}
function coverageFor(f,ch) {
 const cov=dynamicCoverage(f); if(!cov)return null;
 if(cov.channels)return cov.channels[String(ch)]??(num($('channel').value)?null:Object.values(cov.channels).find(v=>v.cells?.length))??null;
 return cov.channel===ch||cov.cells?cov:null;
}
function mapOverlay(side,s) {
 const f=s.frame,ch=num($('channel').value)||num((s.pending??f)?.channel),lastMap=run(side).frames.slice(0,s.index+1).findLast(v=>dynamicCoverage(v)),cov=coverageFor(lastMap,ch);
 const info=$('dynamic-note-'+side),decision=(s.pending??f)?.decision;
 if(info) info.textContent=cov?`频道 ${cov.channel}：方向状态剩余 ${(100*num(cov.remaining_mass)).toFixed(1)}%；实测负反馈 ${cov.negative_measurement_count??cov.negative_points?.length??0} 个停点；${cov.certified?'连续覆盖已认证':'尚未认证'}。色深为该位置剩余方向比例，离散图不作为无源证据。`:(run(side).summary.policy==='dynamic'?(s.frame?.coverage_certified?'剩余未知频道已用连续覆盖证书确认，无未覆盖方向域。':'当前频道没有记录未确认未知域；可选择待探索频道查看探索缺口。'):'旧方案未维护逐频道动态未知域。');
 const raw=$('decision-json-'+side); if(raw)raw.textContent=decision?JSON.stringify(decision,null,2):'此动作没有新的全局候选决策。';
 if(!$('unknown-grid').checked||!cov?.cells?.length)return;
 const svg=$('map-'+side),g=svg.querySelector('g'),layer=document.createElementNS(NS,'g'),spacing=cov.grid_spacing_m??150;
 layer.setAttribute('class','unknown-grid');layer.setAttribute('pointer-events','none');
 const defs=shape(svg,'defs'),clip=shape(defs,'clipPath',{id:'unknown-domain-'+side});shape(clip,'circle',{cx:0,cy:0,r:1800});layer.setAttribute('clip-path',`url(#unknown-domain-${side})`);
 for(const cell of cov.cells){const [x,y,mass]=cell;shape(layer,'rect',{x:x-spacing/2,y:-y-spacing/2,width:spacing,height:spacing,fill:'#8168ac','fill-opacity':Math.min(.36,.04+.31*mass),stroke:'none'})}
 const base=g.children[Math.min(3,g.children.length-1)];g.insertBefore(layer,base);
}
const dynamicOldDraw=drawMap;
drawMap=function(side){const r=run(side),s=dynamicOldDraw(side);mapOverlay(side,s);if(r.summary.completed===false)$('per-'+side).textContent='未完成';if(Object.keys(stats(r,s.index).cleared).length>=16)$('status-'+side).insertAdjacentHTML('beforeend','<br><span class="taggood">已清除公开上限16个源，其余频道无需额外确认。</span>');return s};
const dynamicOldCandidates=renderCandidates;
renderCandidates=function(side,f){
 $('cand-'+side).closest('table').querySelector('thead').innerHTML='<tr><th>动作 / 频道</th><th>评分 / s</th><th>命中概率</th><th>理由</th></tr>';
 dynamicOldCandidates(side,f);const cs=f?.decision?.candidates??[];
 for(const cell of $('cand-'+side).querySelectorAll('.table-note'))cell.textContent=REASONS[cell.textContent]??cell.textContent;
 if(!cs.some(c=>c.cross_source_channels!==undefined||c.cross_source_value_s!==undefined||c.cross_value_s!==undefined||c.cross_saving_s!==undefined||c.known_values!==undefined||c.other_source_value_s!==undefined))return;
 $('cand-'+side).closest('table').querySelector('thead').innerHTML='<tr><th>动作 / 频道</th><th>评分 / s</th><th>跨源价值 / s</th><th>探索价值 / s</th><th>理由</th></tr>';
 $('cand-'+side).innerHTML=cs.map((c,i)=>{const cross=c.cross_source_credit_s??(c.predicted_scans?c.predicted_scans.filter(r=>r.type==='known'&&r.channel!==c.channel).reduce((a,r)=>a+num(r.gross_saving_s)*num(r.credit_weight??1),0):first(c,['cross_source_value_s','cross_value_s','cross_saving_s','other_source_value_s']));return`<tr class="${selectedCandidate(f.decision,c,i)?'selected':''}"><td>${esc(actionLabel(c.kind))} / ${c.channel??'—'}</td><td>${fmt(first(c,['selection_s','score_s']))}</td><td>${fmt(cross)}</td><td>${fmt(first(c,['unknown_credit_s','search_value_s','exploration_value_s','unknown_value_s']))}</td><td class="table-note">${esc(REASONS[c.reason]??c.reason)}</td></tr>`}).join('');
 $('cand-note-'+side).textContent=`动作 ${num(f.index)+1} 执行前的候选；价值单位为启发式预期秒数。完整分解、候选频道和到点后重评记录见下方决策JSON。`;
};
for(const side of ['left','right']){
 const note=document.createElement('p');note.id='dynamic-note-'+side;note.className='explain';$('substats-'+side).after(note);
 const detail=document.createElement('details');detail.innerHTML=`<summary>动态联合决策完整分解</summary><pre id="decision-json-${side}" style="font-size:11px;white-space:pre-wrap;max-height:360px;overflow:auto;background:#f5f8fb;padding:10px"></pre>`;
 $('substats-'+side).closest('section').append(detail);
}
const control=document.createElement('label');control.innerHTML='<input type="checkbox" id="unknown-grid">逐频道未知方向图';$('rays').parentElement.after(control);$('unknown-grid').onchange=render;
for(const row of $('summary-table').querySelectorAll('tbody tr')){
 const failed=(DATA.summary??[]).find(s=>(s.label??label(s.strategy??s.id))===row.cells[0]?.textContent&&(s.failed_count>0||s.mean_per_source_s===null));
 if(failed){row.classList.remove('best');row.cells[1].textContent='未完成，不计完整耗时';row.cells[2].textContent='—';row.cells[3].textContent='—'}
}
render();
"""


def render(data):
    html = render_full(data)
    html = html.replace("第四问 · 完整任务运动历史对照", "第四问 · 动态未知域与跨源停点对照")
    html = html.replace("搜索、定位、清除 · 运动历史对照", "动态未知域与跨源停点 · 完整运动历史")
    html = html.replace("${c.reason??''}", "${REASONS[c.reason]??c.reason??''}")
    html = html.replace("</script></body></html>", EXTENSION + "\n</script></body></html>")
    return html


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path(__file__).with_name("outputs") / "data.json")
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("outputs") / "report.html")
    args = parser.parse_args()
    data = json.loads(args.input.read_text(encoding="utf-8"))
    args.output.write_text(render(data), encoding="utf-8")
    print(json.dumps(dict(report=str(args.output.resolve()), bytes=args.output.stat().st_size), ensure_ascii=False))


if __name__ == "__main__":
    main()

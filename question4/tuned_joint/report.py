"""Build a small, self-contained Q4 motion replay from completed records.

No policy/simulator imports and no scenario execution. Source truth is used only
in the selected offline replay. Missing final inputs produce an explicit error.
Optional PNGs stay relative to the report's figures directory.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path


BASE = Path(__file__).resolve().parent
NAMES = {"old_joint": "最初联合方案", "previous_dynamic": "此前动态方案",
         "free_joint": "上一最佳：自由补点", "tuned_joint": "本轮优化方案"}
FIELDS = {
    "forecast_weight": ("续策路线权重", "权衡当前动作与之后的共同路线代价"),
    "unknown_credit_cap_s": ("未知探索收益上限", "限制未知域收益过度抵扣移动代价"),
    "scan_min_net_s": ("停点测量净收益门槛", "预期节省需超过当前测量及切频费用"),
    "cross_source_weight": ("跨源定位收益权重", "计入一个测点对其他已知源的帮助"),
    "probe_fraction": ("前向探测比例", "控制单源常规候选沿目标方向前进幅度"),
    "lateral_ratio": ("侧向偏移比例", "权衡方位交会几何与额外移动"),
    "revisit_penalty_s": ("重复访问惩罚", "抑制向近期已走过的邻近位置折返"),
}


def read(path):
    path = Path(path)
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            return json.load(stream)
    return json.loads(path.read_text(encoding="utf-8-sig"))


def frozen_parameters(frozen, current):
    if "configs" in frozen and current in frozen["configs"]:
        return frozen["configs"][current]["parameters"]
    for key in ("config", "selected_config", "best_config"):
        if isinstance(frozen.get(key), dict) and "parameters" in frozen[key]:
            return frozen[key]["parameters"]
    if isinstance(frozen.get("parameters"), dict):
        fields = frozen["parameters"]
        if current in fields and isinstance(fields[current], dict):
            return fields[current].get("parameters", fields[current])
        return fields
    raise ValueError("冻结参数格式不明确：需要 configs[策略].parameters、config.parameters 或 parameters")


def locate_history(filename, validation, output=None, seed=None, strategy=None):
    """Resolve original evidence or the byte-identical portable case bundle."""
    original = Path(filename)
    if original.is_file():
        return original
    roots = list(dict.fromkeys([Path(validation).resolve().parent,
                                Path(output).resolve() if output is not None else Path(validation).resolve().parent]))
    for root in roots:
        relative = root/filename
        if relative.is_file():
            return relative
        if seed is None or strategy is None:
            continue
        bundle = root/"case_histories"
        index_path = bundle/"index.json"
        if index_path.is_file():
            entries = [entry for entry in read(index_path).get("entries", [])
                       if int(entry["seed"]) == int(seed) and entry["strategy"] == strategy]
            if len(entries) > 1:
                raise ValueError("便携运动历史索引包含重复的场景/策略")
            if entries:
                entry = entries[0]
                candidate = (root/entry["file"]).resolve()
                if not candidate.is_relative_to(bundle.resolve()):
                    raise ValueError("便携历史索引指向 case_histories 目录之外")
                if candidate.is_file():
                    if hashlib.sha256(candidate.read_bytes()).hexdigest() != entry["sha256"]:
                        raise ValueError("便携运动历史 SHA-256 与保存的证据索引不一致")
                    return candidate
            continue  # An existing index is authoritative for this bundle.
        # A manually copied nine-file bundle may omit its optional index. The
        # caller still verifies seed, strategy and exact recorded mission time.
        candidate = bundle/str(strategy)/f"{int(seed)}.json.gz"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"缺少实际运动历史（原位置与 case_histories 均未找到）：{filename}")


def confirmation_protocol(output, frozen, search, compatibility=False):
    """Read selection scope and manual values from actual confirmation records."""
    path = Path(output)/"confirmation.json"
    if not path.is_file():
        if compatibility:
            return dict(available=False, reason="兼容输入未附本轮 confirmation.json")
        raise FileNotFoundError(f"缺少选择协议及手动消融证据：{path}")
    confirmation = read(path)
    configs, rows, seeds = confirmation["configs"], confirmation["rows"], confirmation["seeds"]
    actual = {(int(row["seed"]), row["strategy"]) for row in rows}
    required = {(int(seed), name) for seed in seeds for name in configs}
    if len(actual) != len(rows) or actual != required:
        raise ValueError("确认数据必须包含各配置的完整共同种子矩阵，不能由缺失行推断选择任务数")
    selected = frozen.get("selected_on_confirmation")
    if selected is None or selected != confirmation.get("selected") or selected not in configs:
        raise ValueError("冻结配置名称与确认选择记录不同")
    values = sorted({float(spec["parameters"]["future_scan_weight"]) for spec in configs.values()
                     if "future_scan_weight" in spec["parameters"]})
    baseline_weight = search["settings"]["base_config"]["parameters"].get("future_scan_weight")
    manual_names = [name for name, spec in configs.items()
                    if "future_scan_weight" in spec["parameters"]
                    and spec["parameters"]["future_scan_weight"] != baseline_weight]
    trial_names = {trial["name"] for trial in search.get("trials", [])
                   if trial.get("proposal", {}).get("kind") != "baseline"}
    return dict(available=True, source="confirmation.json + frozen_parameters.json",
                configuration_count=len(configs), selection_scene_count=len(seeds),
                selection_task_count=len(rows), manual_values=values,
                manual_configuration_count=len(manual_names),
                gp_promoted_count=len(set(configs)&trial_names), selected=selected,
                final_independent_scene_count=len(frozen.get("final_validation_seeds", [])),
                selection_rule=confirmation.get("selection_rule", ""),
                selection_is_adaptive=bool(manual_names))


def compact_history(history, row):
    actual = history.get("commands")
    if not actual:
        raise ValueError(f"场景 {row['seed']} / {row['strategy']} 缺少实际 commands")
    if abs(history["summary"]["time_s"]-row["time_s"]) > 1e-6:
        raise ValueError("运动历史与验证总时间不一致")
    if (history["summary"].get("seed") != row["seed"]
            or history["summary"].get("strategy") != row["strategy"]):
        raise ValueError("运动历史的场景或策略与验证索引不同")
    commands, prior = [], [0., 0.]
    for index, command in enumerate(actual):
        keep = {key: command[key] for key in ("position", "kind", "channel", "time_s", "move_s", "action_s", "response")}
        keep.update(index=command.get("index", index),
                    **{"from": command.get("from", prior)},
                    reason=command.get("reason", ""), phase=command.get("phase", ""))
        commands.append(keep)
        prior = command["position"]
    return dict(summary={key: row.get(key) for key in
                         ("seed", "strategy", "time_s", "per_source_s", "source_count", "cleared", "completed", "costs")},
                commands=commands)


def package_data(output, validation, frozen, checkpoint, selections,
                 current="tuned_joint", reference="free_joint", compatibility=False):
    paths = {"validate.json": Path(validation), "frozen_parameters.json": Path(frozen),
             "bayes/checkpoint.json": Path(checkpoint), "figures/case_selection.json": Path(selections)}
    missing = [f"{name}（{path}）" for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("缺少生成最终报告所需输入：\n"+"\n".join(missing))
    data, frozen_data, search, selection = [read(path) for path in paths.values()]
    if search.get("status") != "complete" and not compatibility:
        raise ValueError(f"贝叶斯搜索尚未完成（状态 {search.get('status')}）；保留真实中间结果，不生成最终报告")
    available = set(data["configs"])
    if current not in available or reference not in available:
        raise ValueError(f"验证结果缺少指定策略 {current}/{reference}")
    strategies = [n for n in ("old_joint", "previous_dynamic", "free_joint", "tuned_joint") if n in available]
    if len(strategies) > 3:
        strategies = [n for n in strategies if n in ("old_joint", reference, current)]
    actual_parameters = data["configs"][current]["parameters"]
    if frozen_parameters(frozen_data, current) != actual_parameters:
        raise ValueError("冻结参数与 validate.json 中实际执行配置不同，不能生成一致性报告")
    protocol = confirmation_protocol(Path(frozen).resolve().parent, frozen_data, search, compatibility)
    space = search.get("settings", {}).get("space", {})
    required = list(FIELDS)
    if not all(name in space and name in actual_parameters for name in required):
        raise ValueError("缺少七个优化参数的实际值或训练范围，不能按默认值填补")
    parameters = []
    for name in required:
        bounds = space[name]
        low, high = (bounds["low"], bounds["high"]) if isinstance(bounds, dict) else bounds
        parameters.append(dict(name=name, label=FIELDS[name][0], description=FIELDS[name][1],
                               value=actual_parameters[name], low=low, high=high, method="GP＋EI"))
    if protocol["available"]:
        actual_weight = actual_parameters.get("future_scan_weight")
        if actual_weight is not None and actual_weight not in protocol["manual_values"]:
            raise ValueError("冻结的未来扫描费权重不在确认记录的实际消融取值中")
        parameters.append(dict(name="future_scan_weight", label="未来未知扫描费权重",
                               description="对假设阴性续策的未来扫描费用降权；只影响预测评分，不改变实际计费",
                               value=actual_weight,
                               values=protocol["manual_values"], method="手动消融（不计入 GP 维度）"))
    finished = [t for t in search.get("trials", []) if t.get("status") == "complete"]
    kinds = Counter(t["proposal"]["kind"] for t in finished)
    grouped = []
    for strategy in strategies:
        rows = [r for r in data["rows"] if r["strategy"] == strategy]
        good = [r for r in rows if r.get("completed") and r.get("per_source_s") is not None
                and r.get("cleared") == r.get("source_count")]
        all_complete = len(good) == len(rows)
        grouped.append(dict(strategy=strategy, label=NAMES.get(strategy, strategy), count=len(rows),
                            completed=len(good), sources=sum(r["source_count"] for r in rows),
                            cleared=sum(r["cleared"] for r in rows),
                            mean_s=sum(r["per_source_s"] for r in good)/len(good) if all_complete and good else None,
                            costs={k: sum(r["costs"].get(k, 0.)/r["source_count"] for r in rows)/len(rows)
                                   for k in ("movement", "measure", "switch", "clear_success", "clear_failure")}))
    chosen = selection.get("cases", selection) if isinstance(selection, dict) else selection
    if not isinstance(chosen, list) or not chosen:
        raise ValueError("case_selection.json 不包含实际选择的案例")
    if len(chosen) != 3 and not compatibility:
        raise ValueError("最终紧凑报告应包含已选择的三个案例")
    cases = []
    row_lookup = {(r["seed"], r["strategy"]): r for r in data["rows"]}
    for item in chosen[:3]:
        seed = item["seed"]
        case = dict(seed=seed, reasons=item.get("reasons", []), runs={}, truth=None)
        for strategy in strategies:
            row = row_lookup.get((seed, strategy))
            if row is None:
                raise ValueError(f"案例 {seed} 缺少 {strategy} 的验证记录")
            filename = locate_history(row["history_file"], validation, output, seed, strategy)
            history = read(filename)
            case["runs"][strategy] = compact_history(history, row)
            truth = [{k: source[k] for k in ("channel", "x", "y", "directional", "orientation")}
                     for source in history.get("truth", [])]
            if not truth:
                raise ValueError(f"案例 {seed} 缺少事后显示所需真值")
            if case["truth"] is not None and case["truth"] != truth:
                raise ValueError(f"案例 {seed} 的策略历史不属于相同场景")
            case["truth"] = truth
        cases.append(case)
    image_paths = []
    for filename in ("bayes_progress.png", "validation_overview.png", "paired_differences.png",
                     *(f"case_{c['seed']}.png" for c in cases)):
        if (Path(output)/"figures"/filename).is_file():
            image_paths.append(f"figures/{filename}")
    return dict(title="第四问 · 贝叶斯优化与实际运动历史", compatibility=compatibility,
                current=current, reference=reference, strategies=strategies,
                labels={s: NAMES.get(s, s) for s in strategies}, summary=grouped, parameters=parameters,
                selection_protocol=protocol,
                structural={k: actual_parameters.get(k) for k in
                            ("adaptive_belief", "elastic_coverage", "near_anchor_probes", "future_scan_weight", "bundle_scans", "replacement_candidates")},
                bayes=dict(status=search.get("status"), completed_trials=len(finished),
                           baseline=kinds["baseline"], sobol=kinds["sobol"], ei=kinds["gp_expected_improvement"],
                           training_scene_count=len(search.get("settings", {}).get("train_seeds", [])),
                           best_training_trial=search.get("best_trial"),
                           best_training_mean_s=search.get("best_mean_per_source_s")),
                paired=data.get("paired", []), cases=cases, figures=image_paths,
                scope="本地合成；各场完整总时间/源数后等权平均，含全部移动、测量、切频和成功/失败清除；近似策略，非全局最优")


PAGE = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>第四问 · 贝叶斯优化与实际运动历史</title>
<style>
:root{--ink:#203c50;--muted:#647d90;--line:#dbe4ec;--blue:#326ba2;--green:#27866b;--paper:#fff;--bg:#f3f7fa}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.65 "Microsoft YaHei","PingFang SC",sans-serif}main{max-width:1480px;margin:auto;padding:30px 26px 45px}h1{font-size:30px;line-height:1.25;margin:8px 0 12px;letter-spacing:.02em}h2{font-size:21px;margin:0 0 12px}h3{font-size:17px;margin:0}p{margin:8px 0 13px}.eyebrow{font-size:12px;font-weight:700;letter-spacing:.15em;color:var(--green)}.muted,small{color:var(--muted)}.badge{display:inline-block;font-size:12px;padding:3px 9px;border-radius:5px;background:#e5f1ec;color:#286852;margin-right:8px}.cards{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:15px;margin:23px 0}.card,.panel{background:var(--paper);border:1px solid var(--line);border-radius:12px}.card{padding:18px 21px}.card strong{font-size:33px;line-height:1.6;letter-spacing:-.03em}.card .unit{font-size:13px;margin-left:5px;color:var(--muted)}.panel{padding:23px;margin:20px 0}.pairnote{padding:10px 14px;background:#edf4f8;border-left:3px solid #8aafc8;border-radius:4px}.controls{display:flex;gap:12px;flex-wrap:wrap;align-items:center}.controls label{display:flex;align-items:center;gap:7px;font-size:13px}.case-select{min-width:260px;max-width:100%;flex:1}.controls select,button{font:inherit;font-size:13px;padding:7px 10px;color:var(--ink);border:1px solid #cbd9e4;background:white;border-radius:7px}button{cursor:pointer}button:hover{background:#eaf2f7}button.primary{background:#316c9f;border-color:#316c9f;color:#fff}.playbar{display:grid;grid-template-columns:auto minmax(80px,1fr) auto;gap:14px;align-items:center;margin:17px 0 9px}.playbar input{width:100%;accent-color:#356f9d}.clock{font-variant-numeric:tabular-nums;font-weight:700;font-size:13px;min-width:145px;text-align:right}.toggles{display:flex;gap:16px;flex-wrap:wrap;font-size:13px;margin:12px 0 16px}.toggles label{display:flex;gap:5px;align-items:center}.comparison{display:grid;grid-template-columns:1fr 1fr;gap:18px}.mapbox{border:1px solid var(--line);border-radius:9px;overflow:hidden;background:#fff;min-width:0}.maphead{padding:12px 14px;border-bottom:1px solid #e7edf2;display:flex;justify-content:space-between;gap:10px;align-items:center}.maphead select{max-width:240px;width:55%;border:1px solid #cbd9e4;border-radius:5px;padding:5px;color:var(--ink)}.maphead strong{font-size:14px;font-variant-numeric:tabular-nums}.map{display:block;width:100%;aspect-ratio:1;background:#fbfdff}.mapfoot{padding:10px 13px;border-top:1px solid #e7edf2;font-size:12px;min-height:108px;line-height:1.8;overflow-wrap:anywhere}.mapfoot .result{font-size:13px;color:#284e68}.legend{display:flex;gap:17px;flex-wrap:wrap;margin:15px 0 4px;font-size:12px;color:var(--muted)}.legend span{white-space:nowrap}.dot{display:inline-block;width:8px;height:8px;background:#337bb0;border-radius:50%;margin-right:5px}.dot.empty{background:transparent;border:1px solid #8e9fb0}.fine{font-size:12px;color:var(--muted);margin-top:12px}.parameter-wrap{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:13px}th,td{text-align:left;padding:10px 12px;border-bottom:1px solid #e2e9ef;vertical-align:top}th{background:#f0f5f9;white-space:nowrap}td.number{font-variant-numeric:tabular-nums;white-space:nowrap}td code{font-size:11px;color:#768fa1}.method{display:grid;grid-template-columns:1fr 1fr;gap:24px}.method ol{padding-left:22px;margin:7px 0}.method li{padding:4px 0}.flow{width:100%;height:auto;margin-top:12px}.gallery{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:18px}.gallery figure{margin:0;min-width:0}.gallery img{display:block;width:100%;height:auto;border:1px solid var(--line);border-radius:7px}.gallery figcaption{font-size:12px;color:var(--muted);margin-top:5px}summary{cursor:pointer;font-weight:700}.warning{border-left:4px solid #ad783a;background:#fff3df;padding:9px 14px;margin:12px 0}footer{margin-top:28px;font-size:12px;color:var(--muted)}
@media(max-width:800px){main{padding:18px 12px}h1{font-size:24px}.cards{gap:9px}.card{padding:12px}.card strong{font-size:24px}.card .unit{display:block;margin-left:0}.panel{padding:15px}.comparison,.method,.gallery{grid-template-columns:1fr}.playbar{grid-template-columns:1fr auto}.playbuttons{grid-column:1/-1}.clock{min-width:120px}.controls{gap:8px}.controls select{width:100%}.case-select{min-width:0;width:100%;flex-basis:100%}.flow{min-height:155px}.parameter-wrap{margin:0 -5px}th,td{padding:8px 7px}.maphead select{width:58%;font-size:13px}}
@media(max-width:460px){.cards{grid-template-columns:1fr}.card{display:grid;grid-template-columns:1fr auto;align-items:center}.card .unit{display:inline}.card small{grid-column:1/-1}.maphead strong{font-size:12px}.toggles{gap:10px}.playbuttons button{padding:6px 8px}}
.case-select select{min-width:0;flex:1}
</style></head><body><main>
<div class="eyebrow">Q4 · LOCAL SYNTHETIC EVALUATION</div><h1>第四问：参数优化与实际运动历史</h1>
<p class="muted">完整任务耗时、参数选择依据，以及慢案例和退步案例的同场景比较。</p>
<div id="compatibility"></div><span class="badge">全部真实动作计费</span><span class="badge">共同场景配对</span><span class="badge">仅本地合成</span>
<div class="cards" id="cards"></div><div id="paired" class="pairnote"></div>

<section class="panel"><h2>实际运动历史</h2><p class="muted">左右共用同一个实际虚拟时刻。移动过程中显示按真实移动时长插值的位置；探测和清除标记仅在动作完成后出现。</p>
<div class="controls"><label class="case-select">场景 <select id="case" aria-label="选择场景"></select></label><label>播放速度 <select id="speed" aria-label="播放速度"><option value="30">30 秒/秒</option><option value="100" selected>100 秒/秒</option><option value="300">300 秒/秒</option></select></label></div>
<div class="playbar"><div class="playbuttons"><button id="start" aria-label="回到起点">起点</button> <button id="previous">上一步</button> <button class="primary" id="play">播放</button> <button id="next">下一步</button> <button id="end">终点</button></div><input type="range" id="time" min="0" max="1" step="0.1" value="0" aria-label="实际虚拟时刻"><output id="clock" class="clock"></output></div>
<div class="toggles"><label><input type="checkbox" id="route" checked>实际轨迹</label><label><input type="checkbox" id="measures" checked>探测点</label><label><input type="checkbox" id="clears" checked>清除结果</label><label><input type="checkbox" id="truth" checked>事后源真值与朝向</label></div>
<div class="comparison"><article class="mapbox"><div class="maphead"><select id="left" aria-label="左侧策略"></select><strong id="left-total"></strong></div><svg id="left-map" class="map" viewBox="0 0 600 600" role="img" aria-label="左侧实际运动历史"></svg><div id="left-info" class="mapfoot"></div></article><article class="mapbox"><div class="maphead"><select id="right" aria-label="右侧策略"></select><strong id="right-total"></strong></div><svg id="right-map" class="map" viewBox="0 0 600 600" role="img" aria-label="右侧实际运动历史"></svg><div id="right-info" class="mapfoot"></div></article></div>
<div class="legend"><span><i class="dot"></i>有信号测量</span><span><i class="dot empty"></i>无信号测量</span><span style="color:#288367">★ 成功清除</span><span style="color:#bd616d">× 失败清除</span><span>◇ 真实源；箭头为发射朝向</span><span>大蓝点：当前位置</span><span>虚线圈：源所在区域边界</span></div>
<p class="fine">只绘制保存的实际命令，不把规划线段当作真实测量。同点多次扫描会重叠显示；详情保留当前动作的频道、反馈和选择原因。真值仅用于此处事后显示。</p></section>

<section class="panel"><h2>参数范围与最终选择</h2><p id="search-note"></p><p id="selection-note" class="pairnote"></p><div class="parameter-wrap"><table><thead><tr><th>参数与选择方式</th><th>搜索范围或消融取值</th><th>最终值</th><th>作用</th></tr></thead><tbody id="parameters"></tbody></table></div><p class="fine">每个提议在相同训练种子上计入完整任务费用；未全清试验受到失败惩罚且不能成为最优配置。训练最优不自动等于确认后冻结的配置。手动扫描费权重消融属于确认选择阶段，不计入 GP 搜索维度。</p></section>

<section class="panel"><h2>目前方案的执行流程</h2><div class="method"><div><ol><li>原点扫描全部频道，建立已发现源与各未知频道的实际观测历史。</li><li>正反馈收缩保守位置多边形；以位置条件的朝向/半径约束积分估计接收概率，保留无信号历史。</li><li>由未知方向缺口、可移动覆盖支持、已知源服务点和沿途共享价值生成连续候选点。</li><li>共同计入移动、测量、切频、清除、后续路线及定位收益，选择一个实际动作。</li><li>每次反馈后重新判断同点值得测量的频道；完成局部补测后再次滚动选点。</li><li>终止依赖全部已知源已清除，且未知频道已有真实连续覆盖证书；或已经清除公开上限 16 个源。</li></ol></div><div><h3>在原方案上的改进动机</h3><p>近端正反馈区域的条件积分容易缺样，因此采用自适应积分与近端锚点候选；它们改善接收概率与定位测点判断。</p><p>固定补点容易造成绕行，因此允许连续调整未来支持点，并用覆盖证书约束其可行性。移动收益同时考虑其他频道，实际经过一个位置并不自动代表完成了测量。</p><p id="structure-note"></p><p class="fine">未来假设的负反馈只用于续策比较，不更新真实无源证据；朝向信息是条件约束与积分，不是已经精确估计出的一个角度。</p></div></div>
<svg class="flow" viewBox="0 0 1200 290" role="img" aria-label="原点扫频，更新联合信息，生成候选，共同评分，单步执行，检查终止并滚动更新"><defs><marker id="flow-arrow" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7Z" fill="#849aae"/></marker></defs><g fill="#eef5f8" stroke="#b8cbd9"><rect x="20" y="35" width="195" height="80" rx="10"/><rect x="250" y="35" width="195" height="80" rx="10"/><rect x="480" y="35" width="195" height="80" rx="10"/><rect x="710" y="35" width="195" height="80" rx="10"/><rect x="940" y="35" width="195" height="80" rx="10"/><rect x="940" y="175" width="195" height="65" rx="10"/></g><g fill="#294d65" font-family="Microsoft YaHei,sans-serif" font-size="18" text-anchor="middle"><text x="118" y="67">原点全频道扫描</text><text x="118" y="95" font-size="13">只记录真实反馈</text><text x="348" y="67">更新联合信息</text><text x="348" y="95" font-size="13">位置、条件朝向、未知域</text><text x="578" y="67">生成自由候选</text><text x="578" y="95" font-size="13">覆盖支持＋跨源共享</text><text x="808" y="67">完整费用共同评分</text><text x="808" y="95" font-size="13">只贪心执行当前一步</text><text x="1038" y="67">执行并在停点补测</text><text x="1038" y="95" font-size="13">按最新反馈重新判断</text><text x="1038" y="202">满足终止条件？</text><text x="1038" y="224" font-size="13">满足：结束；否则：重规划</text></g><g fill="none" stroke="#849aae" stroke-width="2" marker-end="url(#flow-arrow)"><path d="M215,75 H242"/><path d="M445,75 H472"/><path d="M675,75 H702"/><path d="M905,75 H932"/><path d="M1038,115 V167"/><path d="M940,208 H348 V123"/></g><text x="637" y="195" fill="#6a8296" font-family="Microsoft YaHei,sans-serif" font-size="14" text-anchor="middle">有新反馈、范围明显变化或动作已完成：更新再求解</text></svg></section>

<section class="panel"><details><summary>查看可分享的静态统计图与案例图</summary><div class="gallery" id="gallery"></div></details></section>
<footer>本报告仅反映本地合成分布；包含探索、定位、全部移动、测量、切频和成功/失败清除。GP＋EI 是有限预算参数搜索，近似策略没有全局最优保证。交互数据嵌入本页；分享静态图片时请同时保留 figures 文件夹。</footer>
</main><script id="report-data" type="application/json">__DATA__</script><script>
"use strict";
const DATA=JSON.parse(document.getElementById("report-data").textContent);
const $=id=>document.getElementById(id), NS="http://www.w3.org/2000/svg";
const fmt=(v,d=2)=>v==null?"未全清":Number(v).toFixed(d);
const num=v=>Number(v).toLocaleString("zh-CN",{maximumFractionDigits:6});
const esc=value=>String(value).replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
function svgNode(tag,attributes={},text=""){const n=document.createElementNS(NS,tag);for(const [k,v]of Object.entries(attributes))n.setAttribute(k,String(v));if(text)n.textContent=text;return n;}
function put(svg,tag,attrs,text){const n=svgNode(tag,attrs,text);svg.append(n);return n;}
function option(select,value,label){const n=document.createElement("option");n.value=String(value);n.textContent=label;select.append(n);}
let caseIndex=0,time=0,playing=false,lastFrame=0,lastPaint=0;
const activeCase=()=>DATA.cases[caseIndex], run=side=>activeCase().runs[$(side).value];
const maximum=()=>Math.max(run("left").summary.time_s,run("right").summary.time_s);
function pause(){playing=false;$("play").textContent="播放";}
function endedCount(commands,t){let lo=0,hi=commands.length;while(lo<hi){const mid=(lo+hi)>>1;if(commands[mid].time_s<=t+1e-7)lo=mid+1;else hi=mid;}return lo;}
function resultName(c){if(!c)return"尚未完成首个动作";const r=c.response;return({direction:"测得示向度",near:"近距离信号",no_signal:"未接收到信号",success:"清除成功",failure:"清除失败",failed:"清除失败"})[r.measure_result||r.clear_result]||(r.measure_result||r.clear_result||"已记录反馈");}
const reasonLabels={origin_all_channels:"原点全部频道扫描",near_positive_anchor:"近端正反馈锚点探测",dynamic_coverage_completion:"连续覆盖补点",joint_coverage_replacement:"可共同替代未来支持的测点",optical_finite_cover:"有限光学补查",probabilistic_optical_try:"按预期收益尝试清除",certified_clear:"保守几何保证清除",robust_clear_at_stop:"当前位置可保证清除",stationary_known_value:"停点已知源价值补测"};
function reasonName(c){return c?(reasonLabels[c.reason]||c.reason||"常规候选动作"):"尚无反馈";}
function drawMap(side){
  const item=run(side),commands=item.commands,count=endedCount(commands,time),next=commands[count],last=commands[count-1];
  const svg=$(side+"-map");svg.replaceChildren();
  let extent=2100;for(const r of Object.values(activeCase().runs))for(const c of r.commands)extent=Math.max(extent,Math.abs(c.position[0])+90,Math.abs(c.position[1])+90);
  const fx=x=>300+x/extent*274,fy=y=>300-y/extent*274,scale=274/extent;
  const defs=put(svg,"defs",{}),marker=svgNode("marker",{id:side+"-arrow",markerWidth:6,markerHeight:6,refX:5,refY:3,orient:"auto"});marker.append(svgNode("path",{d:"M0,0 L6,3 L0,6Z",fill:"#344e62"}));defs.append(marker);
  for(let q=-2000;q<=2000;q+=1000){put(svg,"line",{x1:fx(q),x2:fx(q),y1:26,y2:574,stroke:"#e8eef3","stroke-width":1});put(svg,"line",{y1:fy(q),y2:fy(q),x1:26,x2:574,stroke:"#e8eef3","stroke-width":1});put(svg,"text",{x:fx(q),y:592,"font-size":10,fill:"#8a9bab","text-anchor":"middle"},String(q));put(svg,"text",{x:5,y:fy(q)+3,"font-size":10,fill:"#8a9bab"},String(q));}
  put(svg,"circle",{cx:300,cy:300,r:1800*scale,fill:"none",stroke:"#adbdcb","stroke-dasharray":"5 4","stroke-width":1.4});
  let robot=[0,0],path="",moving=false,completedClear=new Set();
  for(let i=0;i<commands.length;i++){
    const c=commands[i],start=i?commands[i-1].time_s:0;if(time<start)break;
    const portion=c.move_s>0?Math.min(1,Math.max(0,(time-start)/c.move_s)):1;
    const q=[c.from[0]+(c.position[0]-c.from[0])*portion,c.from[1]+(c.position[1]-c.from[1])*portion];
    if(c.move_s>1e-8)path+=`M${fx(c.from[0])},${fy(c.from[1])}L${fx(q[0])},${fy(q[1])}`;
    robot=q;if(i===count&&time<c.time_s)moving=time<start+c.move_s;
    if(i<count&&c.kind==="clear"&&c.response.clear_result==="success")completedClear.add(c.channel);
  }
  if($("route").checked)put(svg,"path",{d:path,fill:"none",stroke:"#6d91b1","stroke-width":1.65,"stroke-linejoin":"round"});
  for(let i=0;i<count;i++){
    const c=commands[i],x=fx(c.position[0]),y=fy(c.position[1]);let node=null;
    if(c.kind==="measure"&&$("measures").checked){const hit=["direction","near"].includes(c.response.measure_result);node=put(svg,"circle",{cx:x,cy:y,r:hit?2.5:2.15,fill:hit?"#367bb0":"#fbfdff",stroke:hit?"white":"#9cabba","stroke-width":.65});}
    if(c.kind==="clear"&&$("clears").checked){const ok=c.response.clear_result==="success";node=put(svg,"text",{x,y:y+5,"text-anchor":"middle","font-size":ok?18:19,fill:ok?"#278366":"#c36471","font-weight":"bold"},ok?"★":"×");}
    if(node)node.append(svgNode("title",{},`a${c.index+1} · c${c.channel} · ${fmt(c.time_s,1)}秒 · ${resultName(c)}`));
  }
  if($("truth").checked)for(const s of activeCase().truth){
    const x=fx(s.x),y=fy(s.y),opacity=completedClear.has(s.channel)?.44:1;
    put(svg,"path",{d:`M${x},${y-4}L${x+4},${y}L${x},${y+4}L${x-4},${y}Z`,fill:"none",stroke:"#344e62","stroke-width":1.15,opacity});
    put(svg,"text",{x:x+6,y:y-6,"font-size":10,fill:"#344e62",opacity},"c"+s.channel);
    if(s.directional)put(svg,"line",{x1:x,y1:y,x2:fx(s.x+165*Math.cos(s.orientation)),y2:fy(s.y+165*Math.sin(s.orientation)),stroke:"#344e62","stroke-width":1.2,"marker-end":`url(#${side}-arrow)`,opacity});
  }
  put(svg,"rect",{x:296,y:296,width:8,height:8,fill:"#29495f",stroke:"white","stroke-width":1});
  put(svg,"circle",{cx:fx(robot[0]),cy:fy(robot[1]),r:6.8,fill:"#2474ae",stroke:"white","stroke-width":2});
  put(svg,"text",{x:fx(robot[0])+9,y:fy(robot[1])+15,fill:"#236b9e","font-size":11},moving?"移动中":"停点");
  const current=next||last,label=time>=item.summary.time_s?"任务已结束":moving?"移动中":"正在执行停点动作";
  $(side+"-total").textContent=fmt(item.summary.per_source_s)+" 秒/源";
  const feedback=last?resultName(last)+(last.response.svd_deg!=null?` ${fmt(last.response.svd_deg,2)}°`:""):"尚无已完成反馈";
  $(side+"-info").innerHTML=`<div class="result">${esc(label)} · 已完成 ${count}/${commands.length} 个动作 · 成功清除 ${completedClear.size}/${item.summary.source_count}</div>`+
    `<div>当前位置 (${fmt(robot[0],1)}, ${fmt(robot[1],1)}) 米${current?` · ${current.kind==="clear"?"清除":"测量"}频道 c${current.channel}`:""}</div>`+
    `<div>最近完成：${esc(feedback)}${last?`（a${last.index+1}，${fmt(last.time_s,1)} 秒）`:""}</div>`+
    `<div>当前动作原因：${esc(reasonName(current))}</div>`;
}
function draw(){time=Math.max(0,Math.min(time,maximum()));$("time").max=String(maximum());$("time").value=String(time);$("clock").textContent=fmt(time,1)+" / "+fmt(maximum(),1)+" 秒";drawMap("left");drawMap("right");}
function step(direction){pause();const events=[0,...run("left").commands.map(c=>c.time_s),...run("right").commands.map(c=>c.time_s)].sort((a,b)=>a-b);if(direction>0)time=events.find(t=>t>time+1e-5)??maximum();else time=events.filter(t=>t<time-1e-5).pop()??0;draw();}
function animation(now){if(playing){if(lastFrame)time+=(now-lastFrame)/1000*Number($("speed").value);if(now-lastPaint>45){draw();lastPaint=now;}if(time>=maximum()){time=maximum();pause();draw();}}lastFrame=now;requestAnimationFrame(animation);}
function setup(){
  if(DATA.compatibility)$("compatibility").innerHTML='<div class="warning">兼容绘图测试：使用已有记录验证页面功能，本页不代表本轮新验证成绩。</div>';
  $("cards").innerHTML=DATA.summary.map(s=>`<article class="card"><div>${esc(s.label)}</div><div><strong>${fmt(s.mean_s)}</strong><span class="unit">秒/源</span></div><small>全清 ${s.completed}/${s.count} 场 · 已清除 ${s.cleared}/${s.sources} 个源</small></article>`).join("");
  const pairs=DATA.paired.filter(p=>p.first===DATA.current&&p.complete_group&&p.mean_difference_s!=null);
  $("paired").textContent=pairs.map(p=>`${DATA.labels[DATA.current]}相对${DATA.labels[p.second]||p.second}：平均差 ${p.mean_difference_s.toFixed(2)} 秒/源，配对95%区间 [${p.bootstrap95_s.map(v=>v.toFixed(2)).join(", ")}]`).join("；")||"完整费用统计见上方；未提供可用的完整配对区间。";
  DATA.cases.forEach((c,i)=>option($("case"),i,c.seed+" · "+c.reasons.join(" / ")));
  for(const side of ["left","right"]){DATA.strategies.forEach(s=>option($(side),s,DATA.labels[s]));$(side).value=side==="left"?DATA.reference:DATA.current;$(side).addEventListener("change",()=>{pause();draw();});}
  $("case").addEventListener("change",()=>{pause();caseIndex=Number($("case").value);time=0;draw();});
  $("time").addEventListener("input",()=>{pause();time=Number($("time").value);draw();});
  $("start").addEventListener("click",()=>{pause();time=0;draw();});$("end").addEventListener("click",()=>{pause();time=maximum();draw();});
  $("previous").addEventListener("click",()=>step(-1));$("next").addEventListener("click",()=>step(1));
  $("play").addEventListener("click",()=>{if(playing)pause();else{if(time>=maximum())time=0;playing=true;lastFrame=0;$("play").textContent="暂停";}});
  for(const id of ["route","measures","clears","truth"])$(id).addEventListener("change",draw);
  const b=DATA.bayes;$("search-note").textContent=`七个连续参数确实使用 Matérn GP＋期望改善（EI）优化。已完成 ${b.completed_trials} 组：基准 ${b.baseline} 组、Sobol ${b.sobol} 组、GP＋EI ${b.ei} 组，每组使用相同的 ${b.training_scene_count} 个训练场景。下表另外列出手动消融参数；最终值来自实际验证配置，并与冻结文件逐值一致。`;
  const protocol=DATA.selection_protocol;
  $("selection-note").textContent=protocol.available?`选择协议在确认阶段明确扩展：训练选出的 ${protocol.gp_promoted_count} 个 GP 候选与基准比较后，根据这 ${protocol.selection_scene_count} 场的诊断又增加 ${protocol.manual_configuration_count} 个未来扫描费权重消融配置。最终 ${protocol.configuration_count} 个配置 × ${protocol.selection_scene_count} 个共同场景 = ${protocol.selection_task_count} 次完整任务，全部属于自适应参数选择集，不能称作独立最终验证。按确认完整均值选定 ${protocol.selected} 并冻结后，另用 ${protocol.final_independent_scene_count} 个未参与选择的场景作最终独立验证。`:protocol.reason+"；此兼容页不推断本轮选择范围或手动消融结果。";
  const rangeText=p=>p.values?"{"+p.values.map(v=>Number.isInteger(v)?Number(v).toFixed(1):num(v)).join(", ")+"}":num(p.low)+" – "+num(p.high);
  $("parameters").innerHTML=DATA.parameters.map(p=>`<tr><td>${esc(p.label)}<br><code>${esc(p.name)}</code><br><small>${esc(p.method)}</small></td><td class="number">${rangeText(p)}</td><td class="number"><strong>${p.value==null?"本策略未启用":num(p.value)}</strong></td><td>${esc(p.description)}</td></tr>`).join("");
  const s=DATA.structural,features=[];if(s.future_scan_weight)features.push("将未来未知频道扫描费用纳入续策计价");if(s.bundle_scans)features.push("共同评估可替代未来支持点的一组扫频动作");if(s.replacement_candidates)features.push("增加连续插值的替站候选");
  $("structure-note").textContent=features.length?"本轮进一步"+features.join("、")+"，让当前停点的联合价值与未来真正可省掉的费用对应，再用贝叶斯搜索选择权重。":"本次冻结未启用额外结构开关，改进来自已验证结构与参数选择。";
  const figureLabels={"bayes_progress.png":"参数优化实际轨迹","validation_overview.png":"完整费用与共同场景配对","paired_differences.png":"逐场改善与退步"};
  $("gallery").innerHTML=DATA.figures.map(path=>`<figure><a href="${esc(path)}" target="_blank" rel="noopener"><img src="${esc(path)}" alt="${esc(figureLabels[path.split('/').pop()]||"真实运动历史案例")}" loading="lazy"></a><figcaption>${esc(figureLabels[path.split('/').pop()]||path.split('/').pop())}</figcaption></figure>`).join("")||'<p class="muted">尚未附带静态 PNG；交互运动历史数据已完整嵌入本页。</p>';
  draw();requestAnimationFrame(animation);
}
setup();
</script></body></html>'''


def build(output, validation=None, frozen=None, checkpoint=None, selections=None,
          current="tuned_joint", reference="free_joint", compatibility=False, max_bytes=3_000_000):
    output = Path(output).resolve()
    data = package_data(output, validation or output/"validate.json", frozen or output/"frozen_parameters.json",
                        checkpoint or output/"bayes"/"checkpoint.json", selections or output/"figures"/"case_selection.json",
                        current, reference, compatibility)
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"), allow_nan=False).replace("<", "\\u003c")
    page = PAGE.replace("__DATA__", payload)
    encoded = page.encode("utf-8")
    if len(encoded) >= max_bytes:
        raise ValueError(f"报告为 {len(encoded):,} 字节，超过 {max_bytes:,} 字节上限；保留输入，不静默截断动作")
    output.mkdir(parents=True, exist_ok=True)
    destination = output/"report.html"
    destination.write_bytes(encoded)
    return dict(file=str(destination), bytes=len(encoded), cases=[c["seed"] for c in data["cases"]],
                strategies=data["strategies"], command_count=sum(len(r["commands"]) for c in data["cases"] for r in c["runs"].values()),
                compatibility_test=compatibility)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=BASE/"outputs")
    parser.add_argument("--validation", type=Path)
    parser.add_argument("--frozen", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--selections", type=Path)
    parser.add_argument("--current", default="tuned_joint")
    parser.add_argument("--reference", default="free_joint")
    parser.add_argument("--compatibility-test", action="store_true")
    args = parser.parse_args()
    try:
        result = build(args.output, args.validation, args.frozen, args.checkpoint, args.selections,
                       args.current, args.reference, args.compatibility_test)
    except (FileNotFoundError, ValueError, KeyError) as error:
        parser.exit(2, f"无法生成最终报告：{error}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

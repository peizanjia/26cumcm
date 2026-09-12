# 每停点扫频与跨目标滚动规划

## 版本提交（2026-09-12）
- 用户确认当前结果并要求push一次。本次准备提交第三问实现、原始验证证据、报告、交接记录与运行依赖到origin/main；第四问及其文献检索的并行工作不纳入本次提交。
- 提交前沿用已完成的96项测试和冻结源码校验；首次git fetch遭遇GitHub连接重置，推送结果以实际Git返回为准，不将提交准备写成推送成功。

## 用户需求（2026-09-12）
- 用户提供根目录 comparison.html，提出六项改进并要求实际修改代码、分别详细解释：初步试探后重新选源；早期每停点建图；所有停点判断已知/未知频道价值；逐频道保存探索圆；按覆盖缺口动态探索；访问历史与重复邻近点软惩罚。
- 继续以平均每源低于200秒为目标；仅本地合成验证，禁止自动连接官方演练或正式测试。

## 状态与决定
- 新建独立 adaptive_mpc 方案，保留 joint_rollout 冻结基线和用户 comparison.html。
- 每条移动命令之后调用扫描选择函数；主循环每次只执行一个动作并重新比较不同目标与搜索前沿。
- 探索记录为每频道多个1000米检测圆的并集，另保留整格覆盖证书；不把经过位置当作扫描，不用单个大外包圆伪造已覆盖区域。
- 回访为有条件软惩罚；新频道、显著信息收益、保证清除和必要覆盖允许回访。
- 六项代码改动、开发消融、100新场景验证及14历史坏例诊断已完成；当前默认tour250.9197s/源，尚未达到200。完整结果见README；策略核心已冻结，未遗留本轮后台计算。

## 已完成阶段与重要更正
- 已提取 comparison.html 的完整1000场结果：baseline280.3507、terminal277.4953、expectation278.1403、combined275.0694s/源，全部各13060/13060源；14诊断种子单列在 outputs/report_audit/evidence.json。不能把历史2%改进或压力场景8–12%当跨频道耦合的收益上限。
- 实施 parameters/planning/planner/run/benchmark、scanning、exploration，增加每停点扫描、全源候选重选、MEC显著收缩触发、多圆与路径记录、无固定环全域前沿、回访软成本。开启单源基础续策只用于动作评价，真实执行后允许切换。
- 初版20开发场(20272300–20272319)全部完成，但314.8426 vs268.9169s/源退步。曾将有潜在信息的长腿强制分段，后更正为直达与中途点共同竞争；不因单例249s宣布成功。
- v2同20场297.8480s/源，+28.9311；未来覆盖代理遗漏后续清除停点的扫频共享价值。真实时间分解显示最终清除后补扫39.44 vs12.96s/源，约占此次退步91.5%；不代表删除这段就能达到200。
- 140次七组参数消融全部完成（outputs/factor_screen）：降低未知频道扫频密度lean286.6539；关闭早期模式303.0437；关闭回访惩罚286.5106；去除短中继候选294.9250。各组同20场，不能把均值差独立相加。未来覆盖共享代理 anticipated_tail 在另批消融中进行检验。
- 新环境76项适用unittest通过。更正历史测试：旧229.2241黄金成绩漂移不是已证实策略缺陷，当前旧dynamic与joint baseline完整命令一致、均225.8152；用同环境逐动作基线回归替代跨运行时硬编码分数。此前pytest mock失败来自导入路径，本次完整限定名unittest通过。
- 所有试验均使用根目录.venv Python3.12.5，本地合成；未训练ML、未连接官方服务、未commit/push。

## 复现与限制
```powershell
.\.venv\Scripts\python.exe -m question3.global_policy.adaptive_mpc.run --seed 20272000
.\.venv\Scripts\python.exe -m question3.global_policy.adaptive_mpc.benchmark --count 20 --seed-start 20272300 --workers 8 --output question3/global_policy/adaptive_mpc/outputs/new_run
.\.venv\Scripts\python.exe -m question3.global_policy.adaptive_mpc.build_report --input question3/global_policy/adaptive_mpc/outputs/development_v2
```
开发v1的source_hash路径枚举错误导致空集合哈希；v2已修复，后续进一步仅哈希实际策略依赖。开发输出和当时参数保留，不能用当前代码和参数假装复现旧版本。以下为最终冻结版本的实际记录。

## 最终实施与独立验证（2026-09-12，本设备）

- 用户追加原意：分别分析六项影响，最终目标200以下，为此允许调整。实测拒绝默认启用退步版本；改用tour，而非为了满足形式而把全部启发式强加到同一策略。
- `scanning.py`：每个真实停点评估20频道；已知源几何正交筛选加共同情景续策节时，费用只减一次；未知源按真实新增覆盖和剩余缺口筛选。早期6个非原点停点可为明显收缩区域的已知源额外付费建图，日志明确标为`early_geometry_mapping_spend`，不伪称单源VOI为正。
- `exploration.py`：逐频道检测圆并集、访问点和路段；真实扫描才推进覆盖证书。回访15秒软成本，保证清除/有用信息可豁免，不是禁行区。
- `tour_planner.py`：已知源中心与剩余覆盖停点进入同一条开放路线；当前坐标为起点，无强制回原点和固定六边形；每个真实服务动作后扫频再重排。第一步试探无源锁定。预测未来扫频允许提前抵扣覆盖缺口，但只在私有副本上计算，退出只认真实反馈。
- `anticipated_tail.py`：未来已知停点共享覆盖及付费扫描的近似。修正预测980米收缩圈与1000米候选修补不一致导致无进展循环；保留原anticipated_frequent在20272318失败记录，修复后该场12/12完成354.8093s/源及回归fixture。此修复不等于原失败整批重新通过。
- global实验分支保留全源动作估值及450米中途候选；tour采用共路选源+单源连续动作期望，当前并非所有跨源观测反馈的全局最优POMDP。不能把继承参数存在当作对应分支已启用。
- 开发tour使用10个反复筛选场景，回访罚0：251.180s/源；最终候选回访罚15，不能混同配置。六项效应、全部失败和29组开发变体见SIX_DIRECTIONS.md、outputs/six_effects.json。
- 最终验证：种子20272500–20272599，每组100新场景，共400运行。combined274.975713、lean303.916506、sweep_candidate274.130964、tour_candidate250.919655s/源，各1310/1310源、0失败。tour比combined均值少24.056059s/源，配对95%区间[-28.426108,-19.686010]，88/100更快；平均改善8.7484%。tour均值95%区间[242.4039,259.4354]，12/100场低于200，不能据此宣称总体达标。
- 主计分以每场时间/源数再跨场平均。tour成本按同口径为移动173.6383、测量60.3455、切频11.5295、清除5.4064s/源。若其他成本不变，要达200仍需减少约29.3%移动成本；这只是预算换算，不是可达保证。
- 14个原坏案例种子在当前环境配对重跑：全部175/175源，303.776856→266.792181s/源，13快1慢。唯一退步20271390：多252.932秒中209秒为额外测量和切频；末源清除后尾段反而缩短。具体命令/坐标见IMPORTED_CASES.md。这批有选择偏差，不混入100场均值。
- 输入comparison.html SHA256始终为203586707e54edd9fc7754a6e04af16af61f9f2de40ea04b639803e84aefb5ab。最终benchmark源码hash为db0e0021f800f59558c123d77a6540a23d04da5f7e8099cc0f41575a9a021b33，结束后复核一致；source_snapshot.zip内38依赖文件重新计算亦一致。
- `run.py`默认tour，支持--engine tour/sweep/global/combined；有效参数另存selected_configuration.json。CLI复跑20272500的参数、耗时、距离、完成状态与冻结benchmark完全一致；结果243.873084s/源、13/13，单例不作为评价均值。
- 最终96项相关pytest通过（24.11秒）。dynamic测试曾因pytest与unittest包名不同使字符串mock失效，改为patch.object实际导入模块；这是测试定位修正，不改优化器。之前两处旧绝对黄金分数改为同环境完整命令一致性回归，保留漂移说明。
- 新HTML包括100场统计、10个按规则选出的好坏/中位案例；另有原14坏例全量回放。支持逐命令播放、候选点、每停点20频道价值、预测开放路线、实际频道扫描圆。Edge自动检查24个场景、两报告所有变体及多个进度，0页面错误；真值默认隐藏，初始帧无未来覆盖，截图已检查。
- 原始逐场日志、参数、种子、源码快照、配对统计均保存；未提交git，未连接官方演练/正式接口。

## 最终复现命令

```powershell
.\.venv\Scripts\python.exe -m question3.global_policy.adaptive_mpc.run --engine tour --seed 20272500 --output question3/global_policy/adaptive_mpc/outputs/new_demo
.\.venv\Scripts\python.exe -m question3.global_policy.adaptive_mpc.benchmark --count 100 --seed-start 20272500 --workers 8 --variants lean sweep_candidate tour_candidate --output question3/global_policy/adaptive_mpc/outputs/reproduction
.\.venv\Scripts\python.exe -m question3.global_policy.adaptive_mpc.final_audit
.\.venv\Scripts\python.exe -m question3.global_policy.adaptive_mpc.imported_cases_audit
.\.venv\Scripts\python.exe -m pytest question3/global_policy question3/local_sim -q
```

所有输出目录应取新路径，避免覆盖记录。根.venv为本项目Python3.12.5；requirements补充networkx>=3,<4以支持既有路线测试。机器运行库差异可能导致连续优化路径不同，跨设备比较时保留源码快照及版本。

## 未完成目标与下一步

平均每源<200尚未实现。本轮没有证明阈值不可达。下一步应补未知频道发现后对整条路线的分支收益、共同测点对多个源的端点改变，以及在线比较两条完整短续策；当前单源VOI与覆盖代理不能充分表示这些耦合。优先改价值模型，再在新的开发/独立种子上检验，不能继续把这100验证种子用于挑参数后仍称其为独立验证。

# 跨区终点与连续期望动作：本地实验

这是可切换的新方案，保留原dynamic和route_study。接口仍为 `/enter`、`/measure`、`/clear`、`/exit`；本目录运行入口仅连接进程内本地模拟器，没有官方测试模式。

## 当前状态与数据口径

用户已要求“背景继续计算，结束等待”。最终批次和自动报告后处理继续运行；状态快照见 `outputs/validation/background_status.json`，其中时间/完成数为快照而非实时值。后处理会在批次成功完成后自动生成最终HTML；尚未生成前请查看开发/压力案例，不把它们当独立1000场成绩。

- 开发批次：100个场景，种子20269400–20269499，四组均1323/1323源清除。原版276.1383、仅分区271.7573、仅期望274.2518、两项同改269.3916秒/源。这批动作搜索96情景、复核512情景，不作为最终高精度独立成绩。
- 精度检查：24个固定公开状态，主要是第一服务决策；搜索384/复核2048相对96/512，经8192情景共同复核，平均节省0.3727s，最大1.6689s。部分坐标变化大而成本接近；不能宣称每个状态连续最优已收敛。
- 最终冻结配置：搜索384情景、复核2048情景，1024位置粒子；新的1000场种子20271000–20271999，四组共4000次完整运行。运行状态和最终结果以 `outputs/validation/summary.json` 为准；初次撰写时正在执行。
- 开发版12个问题案例HTML：`outputs/development_report/comparison.html`。最终版输出到 `outputs/report/comparison.html`，必须等最终批次结束后生成。
- 开发数据导出100场2517个服务决策状态：`outputs/learning/development_states.jsonl.gz`，只含公开状态与近似成本标签，无真值坐标。未训练机器学习模型。

每场指标是总完成时间/真实源数，再跨场平均；没有将同场多个源当独立样本。官方真实时间预算与本地计算时间、虚拟任务时间分开记录。

## 四组消融

| CLI名称 | 分区关联 | 服务动作 |
|---|---|---|
| baseline | 原版本区前沿/当前位置参考终点 | 原版有限候选rollout |
| terminal | 新的尾部服务与下一入口代价 | 原版有限候选rollout |
| expectation | 原版参考终点 | 连续动作搜索及高样本期望 |
| combined | 新关联终点 | 连续动作搜索及高样本期望 |

用户明确要求分区改与不改都试，故四组全部保留。最终同时比较terminal−baseline和combined−expectation；不因开发均值较好就把分区修改当成普遍保证。

## 分区关联怎样实现

当前点为固定起点。`Planner.terminal_options`先寻找本区尚未纳入当前清理簇的工作；本区尾部已无任务时，查找后续第一个有已知源或未知覆盖缺口的区。

候选入口包含该区已知源中心和根据公开覆盖缺口选出的搜索点。每个入口附带通过剩余已知源、衔接搜索点的开放DP近似后续成本。对当前每个可能末点i，计算

\[
g_i=\min_e\{\|c_i-e\|/5+\widehat C(e)\}.
\]

`Planner.route`使用Held–Karp子集DP，起点代价、边代价和终端代价全部以秒计；只执行第一源服务，再重新建立图。最终无待处理工作时终端代价为0。

这不是强迫最后点落在某条分区边界，而是让清理末点与后续入口经济地衔接。仍保持附近先清、分区有源/未知覆盖双重完成条件，避免为了排序跳过附近已知源。

剩余限制：源中心只是服务结束位置的代理；后续服务暂按移动中心+5s成功清除估计，未知源尚未出现的服务成本未完整建模。下一前沿仍是覆盖启发式候选，尚非全局随机最优路径。DP对当前离散代理图精确，不代表完整任务精确最优。

## 服务期望与滚动执行

`service.choose_action`负责一个已经选定的源。每次真实反馈后重新计算，只执行当次最小估计期望的第一步。

1. r≤20时求最近的保证清除位置：投影到所有顶点r20圆的交集；数值解不满足认证时使用原保守内集点。
2. r>20时比较直接测量、不同前进/侧移量测量、当前位置/圆心/高概率质量位置试清除。
3. 先用多组前进比例及±35/70/140/280m侧移热启动，再用Powell连续优化测量与清除位置；保留原0.5/0.8、70m候选作候选族。
4. 搜索使用384个共同随机情景，候选优胜者使用另一批2048个共同随机情景复核。候选均记录搜索值、复核值、情景标准误、相对直接测量的配对差值/标准误、命中率与预测服务结束位置。
5. 选择复核期望最小者执行。未找到有效概率样本时退回原直接测向续策；到服务步数保护上限时明确失败，不冒充完成。

评估每个候选时都计入移动/5、测量5、必要切频1、清除成功5、miss3，以及随后直到清除完成的基准续策成本。clear成功不切换测量频道；miss后如后续需要测量才计切频。

`kernel.evaluate`中，潜在源坐标仅生成观测结果；后续测点由更新后的公开几何决定。有效观测裁切示向度；no_signal减去1000m圆、miss减去20m圆，并保守凸化。基准续策是在新包围圆中心测量直到可保证清除，同位置观测不独立重复使用。

因此这是**连续第一步样本平均优化+完整基准续策rollout+真实反馈后滚动重算**，不是完整无限分支决策树或任意连续路径全局最优证明。概率状态仍为有限粒子近似；负证据凸化可能丢失孔洞；角度量化似然只通过保守几何近似。数值积分误差、续策近似误差、全局图代理误差要分别讨论。

连续候选测点目前限制在原点半径1800m的搜索域，这是策略搜索范围，不应额外解释为官方接口禁止域外坐标。预测与实际剩余时间的差异，还包含基准圆心续策与真实滚动续策的区别，不能直接当成纯积分误差。

## 函数与参数

| 文件/函数 | 作用 |
|---|---|
| parameters.JointParameters | 所有新增开关和计算精度 |
| planner.Planner.terminal_options | 构造尾部/下区入口与后续代价 |
| planner.Planner.route | 固定起点、带终端代价的精确离散顺序 |
| planner.Planner.service_target | 一步执行、反馈后重算，直到该源清除 |
| service.scenarios | 公开后验位置/接收半径/误差共同情景 |
| service.closest_certified | 最近保证清除点，失败时保守回退 |
| service.choose_action | 连续候选搜索、独立情景复核、最小期望选择 |
| kernel.clip/circle_outer/exclude_disk/mec | 编译几何核；两点和三点包围圆均支持 |
| kernel.evaluate | 各候选全部情景的服务完成成本 |
| benchmark.task/metrics/summarize | 本地运行、压缩全轨迹、折返/预测误差、统计 |
| analyze_results | 两因素配对效应、交互作用、N=10–16分层 |
| build_report.select_cases | 固定规则选差表现案例，不挑最好 |
| export_learning_data | 重建公开状态，导出场景分组的近似教师标签 |

| 参数 | 最终默认 | 意义 |
|---|---:|---|
| linked_terminal | true | 是否使用关联终点 |
| continuous_service | true | 是否使用新期望滚动动作 |
| samples | 1024 | 当前公开后验位置粒子数；baseline/terminal保留原256 |
| search_scenarios | 384 | 连续动作搜索中的积分情景数 |
| validation_scenarios | 2048 | 优胜候选独立抽样复核数 |
| refine_iterations | 24 | 每个连续局部搜索的迭代上限；另有每次120次函数评估上限 |
| refine_finalists | 6 | 搜索排名前列复核数，另强制含直接测量和两类最佳候选 |
| next_entry_limit | 6 | 后续服务/搜索入口计算预算 |

原物理常数、覆盖证书、附近清理500m、8个角向区域等保持不变。没有调任何物理参数。

## 可复现命令

仓库根目录，Python可使用 `D:/ProgramData/anaconda3/python.exe`。

```powershell
python -m question3.global_policy.joint_rollout.run --variant combined --seed 20271000 --output question3/global_policy/joint_rollout/outputs/my_demo
python -m question3.global_policy.joint_rollout.run --variant expectation --seed 20271000 --output question3/global_policy/joint_rollout/outputs/no_terminal_demo
python -m question3.global_policy.joint_rollout.benchmark --count 1000 --seed-start 20271000 --workers 16 --output question3/global_policy/joint_rollout/outputs/validation
python -m question3.global_policy.joint_rollout.analyze_results
python -m question3.global_policy.joint_rollout.build_report
python -m question3.global_policy.replay --input question3/global_policy/joint_rollout/outputs/my_demo
node question3/global_policy/joint_rollout/check_report.cjs
```

已有manifest的输出目录只能以相同种子、配置和策略哈希续跑；修改策略后使用新目录，避免混合版本。开发批次旧配置96/512显式保存在其manifest中，不应直接按新默认往该目录续写。

```powershell
python -m question3.global_policy.joint_rollout.export_learning_data --input question3/global_policy/joint_rollout/outputs/development --variant expectation --limit 100 --output question3/global_policy/joint_rollout/outputs/learning/development_states.jsonl.gz
python -m unittest question3.global_policy.joint_rollout.test_joint question3.global_policy.route_study.test_study question3.global_policy.dynamic.test_dynamic question3.global_policy.test_policy question3.local_sim.test_local question3.local_sim.three_probe.test_certificate
```

## 统计与后续ML

四组固定配对，报告均值差的t区间和场景bootstrap区间、Holm多重比较校正、P90/P95、最差/上尾均值、场景失败率单侧上界及墙钟时间。当前1000场设计针对数秒/源量级差异；不保证任何小差异都能显著，也不证明任意真实分布泛化。

`.json.gz`完整包保存全部公开命令/反馈、候选期望、终点规划及独立评价真值；策略逻辑只访问公开接口。学习导出文件去掉真值，标签明确为近似基准续策成本，不是最优Q真值。训练/验证/最终测试按整场种子分开；本轮独立测试如后续用来调参，下一轮必须另留新的最终测试。

本次不启动机器学习训练。建议先学习近似动作成本和跨区终端代价，保留几何认证与覆盖结束条件。

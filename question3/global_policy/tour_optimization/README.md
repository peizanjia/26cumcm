# 第三问：结构优化与昂贵多场景参数搜索

更新日期：2026-09-12。代码、搜索、最终独立验证和报告均已完成。

后续追问：参与优化的7个参数、真实GP+EI流程、可用于论文的文字，以及新做的100场离线瓶颈/下界分析，见[参数优化与下一步](PARAMETERS_AND_NEXT_STEPS.md)。当前成绩保持不变，下一版联合连续观测与未知发现分支尚未实施。

**本轮选出的最佳配置在100个全新场景中平均242.3597 s/源，同场上一版tour为257.1196 s/源，平均节省14.7599 s/源（5.7405%）。84/100场更快，全部1302/1302个源清除成功。尚未达到平均200以下。**

上一轮报告的250.9197使用20272500–20272599；本轮最终使用20273700–20273799。场景难度不同，比较改善应使用上面的同场257.1196对照，不能把不同批均值直接相减。开发84场的231.2757同样不能冒充最终成绩。

## 最终独立验证

先按每场的总时间/源数计算成绩，再跨100场平均。三组共300次运行、各1302个源，全部完成。选定配置在读取这100场结果之前已保存到[selection.json](selection.json)，参数没有根据这批结果调整。

| 方案 | 平均 s/源 | 平均值95%区间 | P95 s/源 | 全部完成 | 低于200的场数 |
|---|---:|---|---:|---:|---:|
| 上一版冻结 tour | 257.1196 | [249.250, 264.989] | 322.895 | 100/100 | 3/100 |
| 覆盖点坐标优化＋开放路线排序 | 247.0397 | [239.916, 254.164] | 315.618 | 100/100 | 6/100 |
| 结构＋本轮最佳参数 | 242.3597 | [235.614, 249.105] | 298.473 | 100/100 | 9/100 |

| 对比：左−右 | 平均差 s/源 | 配对95%区间 | 左侧更快 |
|---|---:|---|---:|
| structural - previous | -10.0799 | [-13.0313, -7.1285] | 79/100 |
| optimized - previous | -14.7599 | [-18.1533, -11.3666] | 84/100 |
| optimized - structural | -4.6800 | [-7.0930, -2.2669] | 63/100 |

结构本身节省10.0799 s/源；参数优化再节省4.6800 s/源，参数收益的配对区间为[−7.0930,−2.2669]，63/100场更快。本轮最终选定的是84场决选中的bo_11；有限搜索不能证明全局最优。

距离平均200仍差42.3597 s/源，需要在当前均值基础上再降低17.48%。9场低于200不是平均目标达成。全部结果为本地合成模拟，未使用官方演练或正式测试。

## 按要求依次完成的工作

1. **结构分析与实现。** 比较覆盖停点连续微调、路线精确排序、延后未知扫描、局部动作兼顾其他源旁扫，以及坐标/排序组合。14场×6组全部完成；坐标微调+排序组合表现最好，因此将其推进独立验证。之后100场独立结构验证为243.7824 vs253.7583，差−9.9759 s/源。单独精确排序和单独延后扫描没有稳定收益。
2. **昂贵参数优化。** 初步8组拉丁超立方+2组邻域，每组14场；随后实施真正Matérn5/2高斯过程+期望改善EI。复用10组旧结果、补5组初始样本、执行12次GP建议。每组先14场，前5名加结构对照扩至42场，前2名加结构对照扩至84场。相同参数评估多个不同场景块；相同种子的缓存不算新增独立样本。GP阶段实际新增546场模拟、复用140条旧记录。
3. **搜索边界复核。** 对300米初始步长下界补测100/200米，以及更高未知扫描门槛，共4组×14场；均未超过已有候选，未继续晋级。
4. **冻结并独立验证。** 84场决选bo_11为231.2757、未调参结构237.8805；选择后再跑上述100个全新场景，最终242.3597。没有把各阶段均值拼成一个“更大独立样本”。

完整方法、公式与选择依据见[METHODS.md](METHODS.md)。这里使用GP代理和逐级场景预算，参考[GP贝叶斯优化](https://arxiv.org/abs/1206.2944)与[BOHB的预算分配思路](https://proceedings.mlr.press/v80/falkner18a.html)，并非BOHB的KDE原算法。

初步optimize.py曾把未调用的RBF函数误写进算法标签，实际仅LHS+两组确定性邻域；已更正文案并保留[audit_correction.json](outputs/parameter_search/audit_correction.json)。后续[gaussian_process.py](gaussian_process.py)和[bayesian_search.py](bayesian_search.py)才是真正拟合GP并按EI选点。早期聚合文件被第二批覆盖的问题也从原140条逐场文件恢复审计；不隐藏这些更正。

## 哪些代码改变了运动与扫描

[routing.py](routing.py)固定其他节点，寻找某个覆盖停点独自承担的必要方格角点，在覆盖圆交集内最小化它连接前后节点的距离。只接受路程下降、完整覆盖重新校验通过的更新。没有独自承担工作的覆盖点可以删除；已知源中心不在此步骤移动。随后节点不超过13时用Held–Karp求精确开放路线，较大图沿用2-opt。

[planner.py](planner.py)保留每步重排、每停点评估。局部候选评分加入最多两个其他已知源的净扫描节时，乘可调权重后抵扣；扫描费用已在净收益里计算，不会重复扣费。未知扫描门槛、前期几何收缩/接收要求和建图停点数都可调。

这些都是公开状态下的近似：路线使用已知中心与预测覆盖，局部旁扫只重排单源求解器留下的候选；没有完整采样未知源发现后的全部全局分支。预测覆盖只在副本中优化，真实退出仍根据实际清除与真实扫描证书。

## 最佳参数

| 参数 | 原tour/未调参结构版 | 选定值 |
|---|---:|---:|
| initial_step | 550 | 300.000000 |
| unknown_scan_gain | 0.08 | 0.178421 |
| mapping_radius_ratio | 0.6 | 0.738641 |
| mapping_detection_probability | 0.5 | 0.796254 |
| early_geometry_stops | 6 | 7 |
| coupled_scan_weight | 0 | 0.577263 |
| revisit_penalty_s | 15 | 21.987139 |

早期未知阈值与普通阈值同为0.1784213723；覆盖点微调和精确排序开启，延后扫描关闭，坐标优化两轮。完整精度与全部继承配置在[best_parameters.json](best_parameters.json)。上表是联合优化得到的向量，不能把整体收益解释成每个参数的单独因果贡献。

## 时间省在哪里

| 实际费用 s/源 | 上版tour | 仅改结构 | 最佳参数版 | 最佳−上版 |
|---|---:|---:|---:|---:|
| 移动 | 180.454 | 172.074 | 170.091 | -10.363 |
| 测量 | 59.858 | 58.440 | 56.280 | -3.578 |
| 切频 | 11.419 | 11.135 | 10.692 | -0.727 |
| 清除，含失败 | 5.388 | 5.392 | 5.296 | -0.091 |

平均路程从11.5615 km/场降至10.9325 km/场，平均测量从151.04降至141.84次/场。相比只改结构，调参同时略减移动和扫描。末源清除后的平均覆盖尾段从6.0778降至4.8649 s/源；这部分只占整体改善的一小部分。

最坏退步案例20273722仍从311.12升到333.09 s/源，主要多出约24.08 s/源移动；不是每场都改善。回放按新策略最慢、相对退步最大、相对改善最大、中位场景的固定规则选择9例，保留这种失败方向。

## 结果文件与核验

- [最终交互回放](outputs/release_validation/report/comparison.html)：3组策略、逐命令播放、预测路线、候选动作、20频道扫描价值与真实探索圆，真值默认隐藏。
- [优化过程图](outputs/release_validation/report/optimization.png) / [SVG](outputs/release_validation/report/optimization.svg)：小批搜索、逐批扩充与最终独立成绩分开呈现。
- [最终统计审计](outputs/release_validation/audit.json)、[全部300条结果](outputs/release_validation/cases.json)：完整配对差、成本分解、源数分层、失败计数与逐场原始指标。
- [GP搜索记录](outputs/bayesian_search/optimization.json)、[每次EI建议](outputs/bayesian_search/suggestions.json)、[搜索核验](outputs/bayesian_search/search_audit.json)：所有候选与14/42/84场排名。
- [冻结元数据](outputs/release_validation/manifest.json)、[50文件源码快照](outputs/release_validation/source_snapshot.zip)：哈希75645a149b545e6f35f6cdb1d80a8f110b7d9cd8ff3d835fcdcaaf8aece773c3，结束后重建哈希一致，当前源码与快照逐文件一致。
- [实验角色与旧元数据更正](outputs/evaluation_roles.json)：早期结构验证入口的evaluation默认写成development，旧目录final_validation是首批参数的阶段复核；真正最终验证为release_validation。曾误启动的6组结构验证已停止，37条完成输出保留但不混入完整比较。
- [交接记录](AI_INTERACTION.md)：需求、决策、真实结果、更正与后续未完成目标。

105项相关测试全部通过（48.74秒）。最佳参数CLI复跑20273600的参数、时间、路程、测量数及完成状态与决选原记录完全一致。最终HTML经Edge检查9场×3变体、多个进度无脚本错误，初始帧不显示未来覆盖；图像已人工查看。Python3.12.5、NumPy2.5.3、SciPy1.18.1、Numba0.67.0，使用项目根.venv。

## 复现

在项目根目录运行最佳配置，选一个新的输出目录：

```powershell
.\.venv\Scripts\python.exe -m question3.global_policy.tour_optimization.run --seed 20273700 --output question3/global_policy/tour_optimization/outputs/new_demo
```

复现最终三组100场，代码使用同一份冻结配置：

```powershell
.\.venv\Scripts\python.exe -m question3.global_policy.tour_optimization.experiment --specs question3/global_policy/tour_optimization/release_specs.json --count 100 --seed-start 20273700 --workers 10 --evaluation validation --output question3/global_policy/tour_optimization/outputs/reproduction
```

复现昂贵代理搜索（复用本目录保留的先前140条pilot记录）：

```powershell
.\.venv\Scripts\python.exe -m question3.global_policy.tour_optimization.bayesian_search --workers 10 --iterations 12 --output question3/global_policy/tour_optimization/outputs/search_reproduction
.\.venv\Scripts\python.exe -m pytest question3/global_policy question3/local_sim -q
```

本轮没有证明200不可达。继续优化应优先补全未知源发现后的路线分支价值，以及联合生成同时服务多个源的观测位置；当前最终100场已看过，未来继续调参后需要另外保留新种子。

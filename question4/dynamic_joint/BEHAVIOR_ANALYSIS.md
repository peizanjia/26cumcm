# 真实行为与补测效果审计

`behavior_analysis.py` 对已经完成的本地实验日志进行只读分析，不运行策略，不向模拟器发送命令。批次的决策代码哈希核验排除了这个离线分析器，转正文件不会改变冻结策略。

## 分析依据

几何重建函数 `reconstruct_behavior(commands)` 只接受实际命令。每个频道重新创建 `Target`，逐条使用记录的位置和公开回报调用 `observe`；不读取干扰源真值，不使用候选预测结果代替真实测量。真值只由独立物理审计读取，用于核对接收半平面、半径、示向度误差、清除结果及相同种子是否确为同一场景。

旧、新策略的沿途补测统一定义为：`phase=side_known`、执行了真实 `measure`、且该频道在测量前已经被发现。旧策略的 `known_geometry_side_scan` 与新策略的 `reassessed_known_value` 都计入，不能用新字段在旧日志中缺省为零来声称旧策略没有补测。

每条补测保存测量前后的最小包围圆半径、圆心、真实回报、位置、动作编号和实际费用。汇总包括：

- 补测次数、失联次数，以及半径至少减半的次数和占比。
- 从半径大于19.8米变为不超过19.8米的次数。
- 共享点、中途点和细化点实际被执行的次数；只计真实非零移动。
- 选中动作中记录跨源收益的次数，以及同停点真实清除次数。
- 全部实际动作的独立物理、费用、逐频道覆盖与任务完成审计结果。

无信号回报仍然更新接收相容信息，但不会使保守位置凸多边形缩小。这里的“半径减半”是实际几何变化，**不是对整趟任务节省时间的因果估计**。功能执行次数也不能单独证明该功能让总成绩提高；总体耗时仍看同场景完整任务比较及消融。

## 费用与完整性

分析器复用 `audit.audit_history` 的独立核验：移动距离/5米每秒、每次测量5秒、接收切频1秒、真实清除成功5秒/失败3秒。清除不改变接收频道。逐动作时间、累计五类费用、每源账本和最终时间均需一致。

无源频道必须能从自己的真实负观测点建立连续覆盖证明；计划点和离散未知域清空不构成证据。尚未清除真源或动作超限的历史不会产生有效完整成绩。冻结源码哈希与整张实验矩阵检查仍由主 `audit.py` 执行，本行为分析器不替代它们。

## 运行

冻结验证完成后：

```powershell
.\.venv\Scripts\python.exe -X utf8 -m question4.dynamic_joint.behavior_analysis --validation question4/dynamic_joint/outputs/validate.json --output question4/dynamic_joint/outputs/behavior_analysis.json
```

读取开发中已保存的完整历史快照，可限制策略和种子：

```powershell
.\.venv\Scripts\python.exe -X utf8 -m question4.dynamic_joint.behavior_analysis --histories question4/dynamic_joint/outputs/develop_round1/develop_histories --strategies old_joint,cap60_f035,cap60_f100 --seed-start 20285000 --seed-end 20285007 --output question4/dynamic_joint/outputs/develop_round1/early_behavior_audit.json
```

输出JSON保留每场的每条补测证据、策略汇总、配对差异及审计问题。目录快照不意味着整个预定开发矩阵已经完成；正式验证应使用完整的 `validate.json`。

## 已完成的开发快照

20285000–20285007、旧联合与两组新参数共24场、8645条动作，独立审计全部通过。`cap60_f100` 126次沿途已知源补测中99次半径至少减半；旧联合223次中117次减半。该快照仅是开发诊断，不能替代待完成的独立验证。

## 最终独立验证

20288000–20288099、新旧两策略共200场、64435条真实动作的行为与物理审计已完成，全部通过。输出为 `outputs/behavior_analysis.json`；补测比较、五类费用及每源最慢/最大退步/最大改善案例的逐动作解释见 `BEHAVIOR_RESULTS.md`。上节“待完成”描述的是当时的开发阶段。

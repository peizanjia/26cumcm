# 第四问：动态联合补盲、定位清除与贝叶斯参数优化

本目录接续 `free_joint`，保留位置条件朝向/半径信息、逐频道未知方向域、可移动连续覆盖支持、跨源测点评价、到点重评和有限光学补查。新增“付清整组扫描费用后，是否能省去一个未来补盲停点”的联合动作，并对有效参数执行真正的GP＋EI贝叶斯优化。

完整任务的主指标是各场 **全部动作时间÷实际源数** 的等权平均，包含移动、测量、切频及成功/失败清除，要求全部源清除。本目录所有数据均为本地合成，不代表官方演练或正式测试。

已完成真正的7维GP＋EI搜索（33配置），随后在32场上比较7组配置并冻结031。预留新100场每策略1303源全部清除：最初494.09、上一free482.94、本轮481.40秒/源。相对最初改善12.69秒/源，配对95%区间[-19.31,-6.02]；相对上一版改善仅1.54秒/源，区间[-6.54,3.60]，尚不能断言稳定胜过上一版。新方案P90和最慢场也略差于free，规划墙钟增加。完整证据和限制见[RESULTS.md](RESULTS.md)。

[交互运动历史与参数报告](outputs/report.html)约938KB，附3案例×3策略的真实历史和6张正式图；可直接打开HTML，或本地预览`http://127.0.0.1:8778/report.html`。代码入口必须使用本目录[best_config.json](best_config.json)，训练第一名027另行保留。当前是有限搜索预算内按预定选择集选出的配置，不是全局最优证明。

## 阅读顺序

1. [算法文字、公式及流程图](ALGORITHM.md)：状态、候选、评分、实际执行与连续覆盖证书。
2. [本轮参数范围及选择](PARAMETERS.md)：有效维度、固定精度和贝叶斯流程。
3. [483.34秒版本的历史参数审计](../free_joint/PARAMETER_AUDIT.md)：此前是人工消融，没有Bayes；含所有继承参数是否生效。
4. [交互与执行记录](AI_INTERACTION.md)：种子划分、错误更正、实测结果及交接限制。
5. [完整实验结果和差案例](RESULTS.md)：开发、选择、独立验证、费用和下一步方向。

训练采用20295000–07，确认采用20295100–31，参数冻结后才使用20296000–99最终验证。训练和确认都属于选择数据，不把最优训练成绩当独立验证成绩。

## 实现入口

`planner.Planner(command, params)`只接收公开命令回调，不读取模拟器源位置、源数或朝向。选参和绘图程序可在任务结束后读取真值用于评分、物理审计和结果展示。

类默认值是结构消融初值；最终运行必须加载保存的完整参数文件，不能把初值当作Bayes最优向量：

```python
import json
from pathlib import Path
config = json.loads(Path("question4/tuned_joint/best_config.json").read_text(encoding="utf-8"))
if config["policy"] == "tuned_joint":
    from question4.tuned_joint.planner import Planner, Parameters
else:  # 如果确认组没有支持新结构，明确回退到原冻结方案。
    from question4.free_joint.planner import Planner, Parameters
planner = Planner(command, Parameters(**config["parameters"]))
result = planner.run()
```

## 本地复现

```powershell
# 结构开发
.venv/Scripts/python.exe -X utf8 -m question4.free_joint.evaluate --stage develop --variants question4/tuned_joint/outputs/structure_variants.json --seed-start 20295000 --count 8 --workers 8 --output question4/tuned_joint/outputs/reproduce_structure

# 1基准+8 Sobol+24 GP/EI，同8场训练；使用全新目录，恢复时加--resume
.venv/Scripts/python.exe -u -X utf8 -m question4.free_joint.optimization.bayes --base-config question4/tuned_joint/outputs/bayes_base.json --space question4/tuned_joint/outputs/search_space.json --seed-start 20295000 --count 8 --trials 33 --initial 8 --workers 8 --keep-histories --output question4/tuned_joint/outputs/reproduce_bayes

# 重放实际参与选择的7配置、共同32场（选择数据，不是最终测试）
.venv/Scripts/python.exe -u -X utf8 -m question4.free_joint.evaluate --stage develop --variants question4/tuned_joint/outputs/confirmation.json --seed-start 20295100 --count 32 --workers 8 --output question4/tuned_joint/outputs/reproduce_selection

# 冻结方案的共同场景完整验证
.venv/Scripts/python.exe -u -X utf8 -m question4.free_joint.evaluate --stage validate --variants question4/tuned_joint/outputs/frozen_parameters.json --seed-start 20296000 --count 100 --workers 8 --output question4/tuned_joint/outputs/reproduce_validation
.venv/Scripts/python.exe -X utf8 -m question4.dynamic_joint.audit --validation question4/tuned_joint/outputs/reproduce_validation/validate.json --output question4/tuned_joint/outputs/reproduce_validation/audit.json

# 单元/物理/覆盖/选参恢复测试
.venv/Scripts/python.exe -m pytest question4/tuned_joint/test_planner.py question4/free_joint question4/dynamic_joint question4/full_mission question4/zigzag_study/test_study.py -q
```

历史文件含完整实际反馈与运动记录，数量较大，保留本地；Git保存源码、参数、逐场摘要、审计、静态图、小型交互报告以及选中案例的实际压缩历史。摘要中的本机 `history_file` 路径用于本地全量审计，克隆后通过上面的验证命令重新生成全量历史。选中案例的可携副本位于 `outputs/case_histories`，索引包含SHA-256，绘图和报告程序支持读取这些副本。

```powershell
# 使用已提交的摘要及选中案例，重新生成正式图和交互页面
.venv/Scripts/python.exe -X utf8 -m question4.tuned_joint.diagnostics
.venv/Scripts/python.exe -X utf8 -m question4.tuned_joint.report
.venv/Scripts/python.exe -X utf8 -m question4.tuned_joint.summarize
```

`outputs/bayes/best_config.json`是训练集最佳；顶层`best_config.json`才是32场选择后冻结、提交最终测试的配置。两者可以不同。`study freeze`检查登记种子、完整配对矩阵及源码哈希；最终测试历史目录一旦存在，就禁止再次按该目录重新选择。

当前算法仍使用有限候选、局部路线改善与条件全阴性续策；贝叶斯优化仅代表规定范围与预算内的已测试选择，不构成全局时间最优证明。

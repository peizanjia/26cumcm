# 第二监测点的边界感知 Minimax MEC 策略

本目录实现问题二的可解释几何策略。主评价指标统一为两次测向后定位区域的最小覆盖圆半径（MEC radius），旧神经网络不再参与实验。

**提交入口：** [附录代码与支撑材料说明](SUBMISSION.md) · [可独立运行的附录核心代码](appendix_core.py) · [验证记录](VALIDATION.json)。

## 模型

第一次观测后的目标可行域为

\[
\mathcal X_1=B(0,1800)\cap B(S_1,1500)\cap W(S_1,\hat\theta_1,1^\circ),
\]

对任一候选目标位置 \(x\)，与第一次成功接收相容的最小有效半径为

\[
\rho_{\min}(x)=\max\{1000,\|x-S_1\|\}.
\]

完整的可靠接收域定义为

\[
\mathcal C_{\rm rec}=\bigcap_{x\in\mathcal X_1}B(x,\rho_{\min}(x)).
\]

该定义同时考虑 1800 m 目标圆的向外裁切与切向情形。旧 `universal_safe` 仍保留作解析内集和回归测试，但不再用于主策略筛选。

主模型忽略 5 m 局部返回差异：该尺度仅为最小接收半径的 0.5%，在论文末尾作为简化误差说明。圆边界仍使用外接正多边形，使 MEC 评价偏保守。

给定第二点 \(q\) 和可能第二示向度 \(z\)，计算

\[
\mathcal P_2(q,z)=\mathcal X_1\cap B(q,1500)\cap W(q,z,1^\circ),
\]

并令

\[
J_R(q)=\sup_z R_{\rm MEC}(\mathcal P_2(q,z)).
\]

主策略在 \(\mathcal C_{\rm rec}\) 内最小化 \(J_R\)，面积和移动距离只用于同分决策。输出同时包含 5% 近优集合和满足 \(J_R\le20\rm\,m\) 的应用型候选集合。

## 关键实现

- `mechanistic.py`：圆域/角域裁剪、MEC、边界感知可靠接收约束和 Minimax 策略；
- `run_experiment.py`：全局随机案例与边界向外压力案例的三策略配对仿真；
- `safe_region_audit.py`：完整可靠接收域与旧解析 safe 内集的面积和图形比较；
- `convergence_audit.py`：圆多边形边数、未来示向角步长和 safe 角度步长的收敛检查；
- `additional_visualizations.py`：从正式结果生成论文用分布、配对、候选集和最坏几何图；
- `visualization_catalog.py`：生成模型示意、数学原理、算法流程及仿真实验的完整中文图包；
- `tests.py`：几何正确性、直径端点反例和边界恢复测试；
- `ANALYSIS.md`：实验结果与论文使用注意事项。

## 复现

在协作仓库根目录运行；依赖安装和提交材料说明见 `SUBMISSION.md`：

```bash
PYTHONPYCACHEPREFIX=/tmp/q2_pycache python3 -m unittest question2.minimax.tests -v
MPLCONFIGDIR=/tmp/q2_mpl python3 -m question2.minimax.run_experiment \
  --trials 48 --edge-trials 16 --circle-sides 96 --angle-step-deg 2 \
  --safe-angle-step-deg 0.05 --safe-guard-m 0 \
  --output question2/minimax/results_mec
MPLCONFIGDIR=/tmp/q2_mpl python3 -m question2.minimax.safe_region_audit \
  --output question2/minimax/results_mec
python3 -m question2.minimax.convergence_audit \
  --output question2/minimax/results_mec/convergence.json
MPLCONFIGDIR=/tmp/q2_mpl python3 -m question2.minimax.additional_visualizations \
  --results question2/minimax/results_mec
MPLCONFIGDIR=/tmp/q2_mpl python3 -m question2.minimax.visualization_catalog \
  --results question2/minimax/results_mec
```

输出包括：

- `results_mec/summary.json`：聚合指标及配对差异区间；
- `results_mec/trial_results.csv`：逐案例、逐策略结果；
- `results_mec/candidate_regions.png`：中心、向外边界和近切向三类候选区域；
- `results_mec/candidate_points.csv`：可靠候选点、最坏 MEC 半径及 5%/20 m 标记；
- `results_mec/safe_region_comparison.png`：旧 safe 与边界感知完整域的差异；
- `results_mec/comparison.png`：全局与边界压力场景的统一 MEC 指标比较；
- `results_mec/convergence.json`：数值收敛审计。
- `results_mec/mec_distributions.png`：MEC 半径小提琴图及经验分布函数；
- `results_mec/paired_improvements.png`：逐案例配对改进散点；
- `results_mec/worst_case_geometry.png`：最坏后验区域及其最小覆盖圆；
- `results_mec/safe_area_bars.png`、`candidate_set_counts.png`、`numerical_convergence.png`：可靠域面积、候选数和收敛性图。

圆盘使用外接正多边形，对圆域作保守外包络；未来示向角与可靠域约束的有限采样不构成连续域严格认证。概率统计采用额外的均匀合成先验，不是题目给定分布，也不是官方模拟器成绩。

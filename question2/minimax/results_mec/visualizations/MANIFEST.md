# 问题二可视化图包

本目录中的图片均由 `question2.minimax.visualization_catalog` 根据当前正式仿真结果生成，采用中文标签和 220 dpi 输出。

## 建议放入论文正文

| 文件 | 适合位置 | 建议图题 |
|---|---|---|
| `01_problem_geometry.png` | 模型建立开头 | 第一次测向、可靠接收域与第二次后验定位的几何关系 |
| `02_safe_mechanism.png` | 可靠接收域推导后 | 边界感知可靠接收域的约束圆交集解释 |
| `03_mec_theory.png` | MEC 求法证明后 | 最小覆盖圆的两点/三点支撑情形及固定直径法反例 |
| `04_minimax_flowchart.png` | 算法步骤后 | 边界感知 Minimax-MEC 第二监测点算法流程 |
| `07_candidate_landscape.png` | 候选区域定义后 | 移动方向—距离平面上的最坏 MEC 目标景观 |
| `08_worst_bearing_profiles.png` | Minimax 目标解释后 | 不同可能第二示向度下的后验 MEC 半径 |
| `12_success_rate_intervals.png` | 策略比较部分 | 三种策略的 20 m 阈值达标率及 Wilson 区间 |

## 建议用于仿真实验或附录

| 文件 | 用途 |
|---|---|
| `05_sample_scenarios.png` | 展示全局随机样本和边界压力样本如何生成 |
| `06_strategy_actions.png` | 对比三种策略在第一次示向局部坐标系中的动作分布 |
| `09_observed_vs_robust.png` | 检查单次观测结果与条件最坏上界的关系 |
| `10_paired_improvement_histograms.png` | 展示逐案例改进是否稳定，而不只报告平均值 |
| `11_movement_accuracy_tradeoff.png` | 展示移动距离与定位精度之间的权衡 |
| `13_radius_area_relationship.png` | 说明 MEC 半径与多边形面积不是同一指标 |

## 此前生成、仍然可用的图片

上一级目录还包含：

- `candidate_regions.png`：笛卡尔坐标下的完整候选区域、5% 近优集合和 20 m 阈值集合；
- `safe_region_comparison.png`：旧 safe 内集与边界感知可靠域的空间差异；
- `mec_distributions.png`：MEC 半径小提琴图和经验累积分布；
- `paired_improvements.png`：Minimax 与基准策略逐案例配对散点；
- `comparison.png`：均值、P95 和条件最坏半径的柱状比较；
- `worst_case_geometry.png`：三个代表案例的最坏后验区域局部放大；
- `safe_area_bars.png`：旧 safe 保留面积及被边界模型恢复的面积；
- `candidate_set_counts.png`：三类候选集合的离散点数量；
- `numerical_convergence.png`：圆多边形、角度步长和可靠域计算的收敛性。

## 复现

```bash
MPLCONFIGDIR=/tmp/q2_mpl python3 -m question2.minimax.visualization_catalog \
  --results question2/minimax/results_mec
```

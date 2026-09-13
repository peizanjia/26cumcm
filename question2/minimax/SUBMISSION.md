# 问题二附录代码与支撑材料

本次交付采用边界感知可靠接收域内的 Minimax MEC 策略，主指标为两次测向后定位区域的最小覆盖圆半径。所有坐标、距离和半径单位为米。

## 论文附录

`appendix_core.py` 是可单独运行的完整核心代码，仅依赖 NumPy，包含圆/角域裁剪、最小覆盖圆、可靠接收域离散约束、未来示向度扫描以及粗网格和局部细化选点。其算法实现直接从 `mechanistic.py` 导出，去除了旧神经网络权重读取部分，不需要任何模型权重或仓库其他模块。

```bash
python -m pip install -r question2/minimax/requirements.txt
python question2/minimax/appendix_core.py
```

内置示例使用第一测点 `(1700,0)`、示向度 `0°`。论文篇幅有限时，优先摘录 `polygon_min_enclosing_circle`、`reception_supports`、`robust_score`、`MechanisticPolicy.decide`，其余辅助函数随支撑材料提供。修改主实现后，运行 `python -m question2.minimax.export_appendix` 重新导出，避免手工维护两套算法。

程序接口：

```python
import numpy as np
from question2.minimax.mechanistic import MechanisticPolicy

policy = MechanisticPolicy(circle_sides=96, angle_step_deg=2.0,
                           safe_angle_step_deg=0.05, safe_guard_m=0.0)
# __call__ 接收角度制，返回第二点坐标。
qx, qy = policy(1700.0, 0.0, 0.0)
# decide 接收弧度制，并返回候选点、最坏半径及近优标记。
decision = policy.decide(np.array([1700.0, 0.0]), 0.0)
```

## 支撑材料索引

| 文件 | 用途 |
|---|---|
| `mechanistic.py`、`__init__.py` | 标准模块实现与接口；历史 NN 兼容类不参与当前实验 |
| `appendix_core.py`、`export_appendix.py` | 独立附录代码及其生成器 |
| `run_experiment.py` | 48 组全局随机 + 16 组边界压力案例，三策略配对比较 |
| `safe_region_audit.py` | 边界感知可靠接收域和旧解析内集的比较 |
| `convergence_audit.py` | 圆域、未来示向角及可靠域约束角步长的收敛检查 |
| `additional_visualizations.py`、`visualization_catalog.py` | 正式统计图、模型示意和算法流程图 |
| `candidate_heatmaps.py` | 25 m 网格逐点评分及第二测点候选热力图 |
| `tests.py` | 7 项几何和策略回归测试 |
| `results_mec/summary.json`、`trial_results.csv` | 正式实验配置、汇总指标及 192 行逐策略结果 |
| `results_mec/candidate_points.csv` | 代表状态的候选坐标、最坏半径和近优标记 |
| `results_mec/convergence.json`、`safe_region_audit.json` | 数值收敛与可靠接收域审计数据 |
| `results_mec/candidate_heatmaps/` | 圆心及裁切至约 800 m 两例的网格 CSV、NPZ 缓存、图和 LaTeX 插图示例 |
| `paper_minimax_section.tex`、`mec_theory/` | 模型正文、MEC 命题证明及 TikZ 配图源码 |
| `ANALYSIS.md`、`VALIDATION.json` | 结果解释及本次交付验证记录 |

## 完整复现

在仓库根目录执行，推荐 Python 3.10+。当前交付在 Python 3.9.6、NumPy 2.0.2、Matplotlib 3.9.4 上验证；结果不依赖 GPU 或 PyTorch。

```bash
python -m pip install -r question2/minimax/requirements.txt
python -m unittest question2.minimax.tests -v
python -m question2.minimax.run_experiment --trials 48 --edge-trials 16 --seed 20660911 --circle-sides 96 --angle-step-deg 2 --safe-angle-step-deg 0.05 --safe-guard-m 0 --output question2/minimax/reproduced
python -m question2.minimax.safe_region_audit --output question2/minimax/reproduced
python -m question2.minimax.convergence_audit --output question2/minimax/reproduced/convergence.json
python -m question2.minimax.additional_visualizations --results question2/minimax/reproduced
python -m question2.minimax.visualization_catalog --results question2/minimax/reproduced
python -m question2.minimax.candidate_heatmaps
```

热力图脚本固定使用本目录 `results_mec/candidate_heatmaps`，已有 NPZ 缓存时直接重绘；若要重新评分，先将该目录中的 `center.npz` 和 `clipped_800m.npz` 移至备份目录，再运行。中文图片重绘需要系统安装中文字体，例如 Noto Sans CJK SC 或微软雅黑。字体查找失败不会改变数值结果。

主实验的完整复现输出写入 `reproduced/`，便于与已提交的正式结果比对；机器速度不同，`decision_time_ms` 和 `elapsed_s` 不要求一致。

## 论文表述边界

- `results_mec/` 为当前 MEC 指标结果；仓库原有 `results/` 和 `RADIUS20_COMPARISON.md` 属于历史版本，不作为本次论文配对比较依据。
- 圆的外接多边形保守外包真实圆域；未来示向度和可靠接收约束仍采用有限角采样，因此不能把整体数值结果称为连续域严格上界证书或全局最优解。
- 5% 近优标记实际阈值为 `max(1.05 * 最优半径, 最优半径 + 0.05 m)`；`J_R <= 20 m` 是离散条件最坏指标门槛，不是未来观测成功概率。
- 均匀目标/误差/接收半径是额外合成先验；本次材料不包含官方演练或正式测试成绩。主模型忽略 5 m 特殊返回规则。

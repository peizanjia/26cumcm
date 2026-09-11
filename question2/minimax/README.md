# 第二监测点：集合成员与条件 minimax 策略

协作者最新 `radius20` 网络的临时对比见 `RADIUS20_COMPARISON.md`；在协作者仓库根目录运行主实验后可用
`python -m question2.minimax.compare_new_radius_nn` 重新生成 JSON 和对比图。若将本目录单独放在工作区根目录，则把模块前缀改为 `question2_mechanistic`。

本目录实现一个不依赖神经网络的第二监测点策略，并在相同合成场景中与协作者仓库的原始神经网络比较。

## 模型

第一次测向后的可行域为

\[
P_1=B(0,1800)\cap B(S_1,1500)\cap W(S_1,\hat\theta_1,1^\circ).
\]

候选点必须通过闭式充分条件保证第二次能够接收信号。以第一示向方向为极轴，令第二点相对第一点的极坐标为 \((s,\beta)\)，则

\[
s\le1000,\qquad s\le2000\cos(|\beta|+1^\circ).
\]

对候选点 `q` 和所有与第一次观测相容的未来示向度 `z`，计算

\[
P_2(q,z)=P_1\cap B(q,1500)\cap W(q,z,1^\circ),
\]

并以

\[
J_D(q)=\sup_z\operatorname{diam}P_2(q,z)
\]

为主目标。面积仅用于直径并列时的次级判据。程序先进行确定性粗网格搜索，再在最优点附近细化；圆用外切正多边形近似，因此得到的是保守外包络。

## 复现

项目只依赖 NumPy 和 Matplotlib。对比实验直接读取仓库已有的
`question2/黑箱思路/checkpoints/best.pt`；该 PyTorch 权重由一个很小的只读解析器转换为 NumPy 数组，不需要安装 PyTorch。

```bash
python3 -m unittest question2.minimax.tests
MPLCONFIGDIR=/tmp/matplotlib python3 -m question2.minimax.run_experiment --trials 96
python3 -m question2.minimax.convergence_audit
```

结果写入 `question2/minimax/results/`：

- `summary.json`：聚合指标、置信区间和实验配置；
- `trial_results.csv`：逐场景、逐策略的配对结果；
- `candidate_region.png`：代表场景的候选点、近优区域和最优点；
- `candidate_points.csv`：代表场景中满足可靠接收条件的离散候选集合及其最坏指标；
- `comparison.png`：平均面积、尾部直径和稳健直径上界对比。
- `convergence.json`：圆离散和示向角扫描分辨率审计。

完整的结果解释见 [`ANALYSIS.md`](ANALYSIS.md)。

## 对比口径

- `area_m2`、`diameter_m`：成功收到第二次信号时，同时使用第二点的 1500 m 最大接收距离约束；
- `legacy_area_m2`：复现仓库旧实验的几何口径，不使用第二点接收圆；
- `safe_certified`：是否具有对全部第一可行位置的接收保证；
- `conservative_worst_diameter_m`：测量前最坏直径。不能保证接收的策略按“无信号后保留第一次可行域”保守计分。

所有数值都是题目约束下的本地合成仿真，不是官方模拟器结果。

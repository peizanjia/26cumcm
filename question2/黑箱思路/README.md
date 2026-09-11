# 第二问 · 黑箱策略网络

**当前默认方案：[半径接近20、正交引导、策略与可行域](radius20/README.md)。** `run.ps1` 默认运行新半径目标；旧面积流程使用 `-Objective area`。新 NN 点接口是 `radius20.policy.pi`，可行候选策略接口是 `radius20.policy.candidate_strategy`。下文为历史面积实验。

已经完成条件概率建模、4 个网络并行训练、独立测试、候选区域计算和几何核验。详细推导及限制见 [分析报告](分析报告.md)。

后续追问已补充 [对称双分支、实际训练机制、公平基准与固定真值热力图](symmetry/README.md)。下表保留首版测试；其中 750/375 是未优化的手设基准，新方案已经单独优化固定步长并使用新测试集。

## 当前结果

默认合成模型下，8,192 个独立测试状态、每状态 256 个条件样本：

| 策略 | 平均剩余面积 / m² |
|---|---:|
| 原始神经网络 `pi` | 503.42 |
| 带接收保证的 `safe_pi` | 503.68 |
| 后验矩解析策略 | 502.76 |
| 自适应几何基准 | 570.90 |
| 固定前进 750 m、侧移 375 m | 655.25 |

黑箱思路可行，但本次网络未胜过后验矩解析策略。所有数字来自本地合成实验，不能写成官方模拟器成绩。

## 直接调用

在仓库根目录运行，角度单位为度、坐标单位为米，正东为 0°、逆时针增加：

```python
from question2.黑箱思路.policy import pi, safe_pi

print(pi(0, 0, 0))
# 约 (974.64, -271.97)

print(safe_pi(0, 0, 0))
# 约 (965.40, -260.78)
```

`pi` 的公开输入仍然是 `x,y,theta` 三个量；内部将第一测点旋转到示向度坐标系，网络使用两个无冗余的旋转不变量，最后还原世界坐标。

模型适用范围：第一测点在半径 1800 m 圆域内，且已经得到合法示向度。第二测点可以在圆域外。第一测点在圆外时，本版网络会明确拒绝，不能视为经过训练的输入；应使用下述条件候选搜索或另行扩充训练分布。

`safe_pi` 额外提供几何接收保证：在题目硬条件与误差界下，第二点能够收到信号或返回 `near`；`near` 时可光学精确定位。它不是“所有目标位置都必定返回示向度”的保证，也不保证只靠两次检测就能在 20 m 内清除。

## 复现实验

安装适合本机 CUDA 的 PyTorch，再安装目录依赖。本次环境：Python 3.13.5、PyTorch 2.9.0+cu126、RTX 4060 Laptop GPU。

```powershell
python -m pip install -r 'question2/黑箱思路/requirements.txt'

# 全流程：训练 → 独立评价 → 四组候选区域 → 图片 → 测试
& './question2/黑箱思路/run.ps1' -Task all

# 只运行某一步
& './question2/黑箱思路/run.ps1' -Task test
& './question2/黑箱思路/run.ps1' -Task evaluate
```

若默认 `python` 指向不同环境，可指定 `-Python 'D:\ProgramData\anaconda3\python.exe'`。脚本为本机 Anaconda/PyTorch 的 Intel OpenMP 冲突选择 MKL 顺序后端，不使用允许重复加载 OpenMP 的绕过开关。

任意给定初始观测的候选区域：

```powershell
$env:MKL_THREADING_LAYER='SEQUENTIAL'
python -m question2.黑箱思路.candidates --x 500 --y 300 --theta 65 --size 101 --samples 2048
```

该命令输出每个网格点的坐标、面积估计、标准误、失联概率、5% 次水平集标志与接收证书；计算窗口和网格分辨率均显式记录。原有训练权重不会改变。

## 文件导航

| 文件 | 用途 |
|---|---|
| `分析报告.md` | 题意、概率模型、目标函数、推导、实验和局限 |
| `environment.py` | 独立先验生成、首次观测条件化、精确圆域条件抽样 |
| `geometry.py` | CPU/GPU 凸多边形裁切、可微面积、接收证书 |
| `circle_reference.py` | 独立连续圆域积分，用于检验圆多边形近似误差 |
| `objective.py` | 对隐藏接收半径解析积分后的损失 |
| `policy.py` | 网络、`pi`、`safe_pi` |
| `analytic_baseline.py` | 后验距离矩解析基准 |
| `train.py` | 4 个网络合并批次训练与验证集选模 |
| `evaluate.py` | 未见状态、配对比较、敏感性与独立几何审计 |
| `candidates.py` | 候选区域网格及独立样本复核 |
| `checkpoints/best.pt` | 已训练并选择的网络权重 |
| `results/config.json` | 训练配置、环境与随机种子 |
| `results/training.csv` | 所有验证检查点 |
| `results/evaluation.json` | 完整测试结果与置信区间 |
| `results/heldout_state_results.npz` | 逐状态均值和网络动作，方便配对复核 |
| `results/candidates/` | 四个场景的逐网格 CSV、元数据 JSON 和绘图 NPZ |

![训练和比较](results/training_comparison.png)

![候选区域](results/candidate_regions.png)

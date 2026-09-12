# 路线算法、补测价值与理论下界研究

## 2026-09-12 · 四项追问审计（最新）
- 用户追问分区出口/入口关联、300回合损失、最小包围圆更新、密度积分期望路径、连续试探距离以及ML样本量。
- 核验当前代码与输出，新增 `FOLLOWUP_ANALYSIS.md` 和 `outputs/followup_statistics.json`；未修改策略、未增加模拟、未训练、无官方连接。
- 更正：300场属于旧单目标，不含分区。跨区关联尚未联合优化；当前参考终点是本区前沿或当前位置。无前沿自由终点20场差−0.5413，95%配对t区间[−5.1561,4.0735]，不能声称主要损失已量化。
- 包围圆在有效新direction/near后重算（两点直径/三点支持均考虑）；no_signal/miss只改排除区和样本，保守凸包圆不缩。dynamic不使用r60硬阈值，rollout已有期望比较，但候选有限，续策未完整利用新增负证据。
- 连续动作建议为前进ell与侧移s联合SAA优化，保留清除/测量分支与跨区终端价值；尚未实现。后验一般不均匀，当前已有接收半径可行区间长度权重。
- 旧单目标300场斜前比直走配对差−1.9431s，95%区间[−3.0506,−0.8356]；多源30场自由终点差−2.9596，区间[−7.4336,1.5145]。外推300场自由终点差值半宽约1.36s/源，外推不是新实测。
- 复算环境D:/ProgramData/anaconda3/python.exe；scipy.stats.t.ppf(.975,n−1)乘原summary的SE；功效规划ceil(((norm.ppf(.975)+norm.ppf(.8))*s_d/3)^2)，零失败单侧上界1−.05^(1/n)。输入路径与假设见分析文档。
- 下一步建议：跨区连接配对消融→后验续传和连续测点→冻结策略扩大独立测试→学习动作价值/终端价值。这些为计划，不是已实现结果。

## 用户需求（2026-09-12）
- 估计每源时间优化目标与理论极限，区分数学界和工程目标。
- 更换局部路径规划算法进行实际本地比较，解释参数意义。
- 已知ρ>20频道在每个停点判断补测收益；未知频道继续按覆盖区域选点。
- 比较内圈先清再外圈、当前单圈扫掠及其他整体路线。
- 只进行本地仿真；不连接官方演练，不用正式次数。

## 方案与状态
- 保留dynamic基线，新实验放本目录；所有策略仅访问公开command接口。
- 下界计算允许离线读取真值，独立于实际策略：起点至r20目标圆的开放Hamilton路径边代价放宽，Held–Karp动态规划精确求放宽模型。
- 计划比较插入+2-opt、最近邻、圆心开放路径精确DP；整体单扫掠、内圈先行、自由选择下一分区；补测启发式与样本VOI。
- 先固定种子做算法筛选，再用新的独立种子验证选中方案；筛选结果不冒充独立泛化表现。
- 本地实现、筛选、独立验证和图形检查已完成。原dynamic默认算法未替换，没有启动连续调参或官方测试。

## 初步已执行
- 真值离线下界100个种子20263000–20263099：开放r20邻域边距离放宽，均值136.412203s/源；固定原点20频道扫描增加119s/局，下界均值145.789065s/源。全知圆心开放路线可行解均值144.093474s/源，属于全知问题的可行上界，不能当在线问题的上界。
- 7项新测试通过，DP对小规模全排列核验、r20邻域界、重复测点无收益、无信号负收益及策略完成性。
- demo种子20260911仅用于调通：基线229.2241、最近邻227.8759、精确DP229.2241、内圈800为297.7530、自适应分区232.2304、每停点VOI227.4590s/源。不能由此单场选最终算法。
- 开始20个固定筛选种子20263000–20263019的配对算法比较；另测试最近邻+VOI组合。基线复用dynamic保存的相同种子结果（该子集267.1750s/源），不是拿全100场279.28直接比较。

## 筛选完成与独立验证计划
- 20场、每版本272/272源清除。平均s/源：基线267.1750；最近邻260.3149；固定终点精确DP267.6666；自由终点精确DP261.2645；内圈800为329.0755、内圈1100为326.0282；自适应分区266.5221；原停点VOI266.6429；每停点VOI266.8926；最近邻+每停点VOI261.0495。
- 这些是筛选结果，并不证明小幅差异稳定；局部路线每次候选通常很少（20场最近邻272次选路，只有82次多于一个目标，最多5个）。
- 在看独立结果前确定验证：30个新种子20269200–20269229，比较基线、最近邻、自由终点DP、最近邻+VOI、内圈800。未将这批用于调整参数。
- 加入更强但依赖当前完成条件的下界：逐个无源频道用1000m圆覆盖1800m圆场地，面积比3.24意味着至少4次测量；扣除原点1次后每个无源频道至少再测3次。因此在固定20频道原点扫描和完整无源覆盖条件下，100场下界均值154.428214s/源。该限制性下界不适用于允许按源数上界提前结束的其他策略。

## 追加用户方向：图论与静态分块（同轮）
- 用户认为demo局部仍有绕圈，提出本区未知先扫描、再静态清理，要求经典图算法并讲解全局源之间的联系。
- 追加MST前序+2-opt、Christofides开环改造、模拟退火；另设先扫描完整本区后冻结清理顺序、全体已知目标主顺序指导局部、全体已知目标开放图优先清理三个结构对照。
- 先扫描再清理是实验性替代顺序，允许扫描阶段暂缓附近清理；不修改原dynamic安全完成条件或默认策略。只冻结服务顺序，单个未知位置仍须测向/清除反馈更新，不假装真坐标已知。
- 新增相同冻结公共图的静态算法审计，将静态长度最优与在线总时间分开核对。原30场独立验证继续，追加图算法用原20筛选种子；不得把追加筛选当独立验证。

## 最终已执行结果
- 17种算法/结构（含基线）在相同20筛选场景均272/272源清除。完整表 `outputs/comparison.json`、`screening_table.md`；经典图算法s/源：插入+2-opt267.1750、最近邻260.3149、固定终点DP267.6666、自由终点DP261.2645、MST+2-opt266.8250、Christofides开环270.0196、退火267.5694。
- 用户追加结构：块内先扫描后冻结顺序272.5427（+5.3676）；全局主顺序指导局部264.1074（−3.0676）；全部已知源优先全局图303.1339（+35.9589）；无前沿自由终点266.6337（−0.5413）。全部是20场筛选，未声称独立验证。
- 独立30种子20269200–20269229：每方案371/371源。基线291.366979；最近邻288.175383（差−3.191596，SE2.283434）；自由终点DP288.407428（差−2.959551，SE2.187554）；最近邻+VOI286.241634（差−5.125345，SE3.472789）；内圈800为357.212835（差+65.845856，SE4.851529，30/30都更慢）。小幅改善候选95%区间跨0，不宣布稳定胜出，不覆盖原默认。
- 静态审计20个基线场景、88个多目标公共图：插入+2-opt、精确DP、MST+2-opt、退火全部88/88同长；Christofides80/88，平均多7.436499m；最近邻51/88，平均多67.776151m。24图起点=参考终点，揭示无前沿时虚拟返回成本。最初3场15图审计已扩展；扩展时曾遇numpy.int64 JSON序列化失败，转成Python int后重跑，最终88图已保存。
- 已知补测VOI可在每个定位停点评估，但当前是单源安全续策近似，精评预算默认6频道×6情景。没有证明全局真实VOI，也没有观察到稳定总体提速。
- 57项相关测试通过（本目录新增11项＋既有46项）。检查范围含精确DP对枚举核验、邻域下界、图算法有效排列与退火保留最好解、静态分块/全局完成性、自由终点、信息价值、既有几何与协议。
- 比较页面 `outputs/comparison.html`：17种实际路线、6种静态算法、88张冻结图、两个同源场景切换、原轨迹叠加，浏览器无JS错误，截图已目视检查。VOI单场回放已补净收益表。

## 新增/修改文件
- 本目录 `routes.py`、`graph_algorithms.py`、`planner.py`、`parameters.py`、`information.py`、`bounds.py`、`benchmark.py`、`static_audit.py`、`run.py`、`test_study.py`、`build_report.py`、`report_template.html`、`check_report.cjs`、`example_parameters.json`、`README.md`、`PARAMETERS.md`。
- 上一级replay.py/replay_template.html兼容实验参数，展示随阶段变化的扫掠方向、补测VOI表。
- root/question3/global_policy交互索引同步；未修改dynamic原策略，不commit/push。

## 可复现命令
仓库根目录，Python为D:/ProgramData/anaconda3/python.exe；完整命令见README。
```powershell
python -m question3.global_policy.route_study.run --variant nearest --seed 20260911 --params question3/global_policy/route_study/example_parameters.json
python -m question3.global_policy.route_study.benchmark --reuse-baseline
python -m question3.global_policy.route_study.benchmark --reuse-baseline --variants mst_2opt christofides annealing block_first global_guided global_graph --output question3/global_policy/route_study/outputs/graph_screening
python -m question3.global_policy.route_study.benchmark --count 30 --seed-start 20269200 --variants nearest exact_open nearest_voi inner800 --output question3/global_policy/route_study/outputs/holdout
python -m question3.global_policy.route_study.bounds
python -m question3.global_policy.route_study.static_audit
python -m question3.global_policy.route_study.build_report
python -m unittest question3.global_policy.route_study.test_study question3.global_policy.dynamic.test_dynamic question3.global_policy.test_policy question3.local_sim.test_local question3.local_sim.three_probe.test_certificate
```

## 结论与限制
- 优化目标应称时间下界；当前100场保留原点扫描+无源覆盖条件的乐观下界154.43s/源，不是已可达策略。建议工程目标先250、再230、200挑战，均未由本次独立验证实现。
- 局部图过小、虚拟终点、信息变化和未知前沿的联系比换更复杂静态TSP算法更值得研究；两阶段完整内圈策略本次明显更慢。
- 后续可研究全局粗图＋定位足够好小簇精排，将未知前沿与所有已知源服务共同计入短时域价值。此联合框架尚未完整实现，不写成已验证最优方案。
- 所有成绩为本地假设下合成场景。不同组源数和位置分布不同，20场267.18、原100场279.28、独立30场291.37均是各自原版基线，不能相互直接作算法增益。

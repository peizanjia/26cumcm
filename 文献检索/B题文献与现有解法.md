# 2026 国赛 B 题「无线电干扰源快速自动定位与清除」文献与现有解法检索

更新日期：2026-09-12。本文由联网检索汇总，供问题 1–4 的建模与论文写作参考。

> 2026-09-11复核说明：上行日期为旧文件原标记，保留用于追溯。本文件曾误写“以定位区域直径为直径的圆必然覆盖区域”，已在第6节更正；等边三角形即可反证。90°最优只适用于相应固定距离及噪声条件，不能作为全局测点最优结论。本文件其余旧书目未在本轮全部重核，第三问请优先使用[专题核验结果](第三问_主动搜索定位清除_专题检索.md)。

## 0. 重要前提（先读）

- 本题为 **2026 年新题**。本轮未核实针对本题的官方解答或已发表优秀论文，不能据此断言不存在。原题建议2026-09-13 15:30前完成正式测试，17:30之后不能启动新测试；已开始测试仍可正常完成。
- 最接近的「现有解法」是 **2022 年高教社杯国赛 B 题「无人机遂行编队飞行中的纯方位无源定位」**——同样是纯方位（示向度/AOA）交会定位，已有大量优秀论文与开源实现，方法可直接迁移到本题问题 1、2。
- 学术上，本题核心技术链条为：**纯方位被动定位（bearings-only / AOA）→ 交会定位误差区域（GDOP/定位模糊区）→ 最小包围圆（smallest enclosing circle）→ 最优测站/测点布设（optimal sensor placement, FIM/CRLB）→ 移动平台搜索与路径规划（active source seeking / informative path planning）**。下面按问题归类。

---

## 1. 问题 1：交会定位区域直径 + 最小包围圆覆盖

### 1.1 最小包围圆（最小圆覆盖）算法 —— 直径圆能否覆盖的关键结论

- **Welzl, E. (1991). Smallest enclosing disks (balls and ellipsoids).** 最小包围圆的随机增量算法，期望线性时间 `O(n)`。这是「以直径是否为直径作圆、能否覆盖定位区域」的标准判定工具：**多边形的直径 = 最远点对距离；最小包围圆直径 ≥ 多边形直径，且当最远点对同时位于最小包围圆直径两端时二者相等**。
  - 现成实现：CGAL《Bounding Volumes》最小包围圆模块：<http://www-sop.inria.fr/members/Clement.Jamin/CGAL/Td_doc_r3/Bounding_volumes/>
  - NetTopologySuite `MinimumBoundingCircle`：<http://nettopologysuite.github.io/NetTopologySuite/api/NetTopologySuite.Algorithm.MinimumBoundingCircle.html>
  - 中文科普（最小圆覆盖算法原理）：<https://www.kepuchina.cn/article/articleinfo?business_type=100&classify=0&ar_id=201105>
- 参考：StackExchange「多边形能否被以直径 a 的圆覆盖」的几何讨论：<https://math.stackexchange.com/questions/4347661/prove-that-a-polygon-with-perimeter-2a-can-be-covered-by-a-circle-with-diamete>

### 1.2 交会定位误差区域 / 定位模糊区（问题 1 的多边形定位区域来源）

- **测向交会定位跟踪的保精度空间区域划分**，空军工程大学学报(自然科学版)，2006 年 03 期。讨论交会定位的可保证精度空间区域，与「定位区域直径」直接相关。链接：<https://wap.cnki.net/touch/web/Journal/Article/KJGC200603010.html>
- **测向定位中若干问题的探讨**，无线电工程，2001 年 S1 期。基础交会定位与误差分析。<https://wap.cnki.net/touch/web/Journal/Article/WXDG2001S1040.html>
- **基于测向定位的算法研究**，现代电子技术，2004 年 04 期。<https://wap.cnki.net/touch/web/Journal/Article/XDDJ200404017.html>（万方：<https://d.wanfangdata.com.cn/periodical/ChdQZXJpb2RpY2FsQ0hJU29scjlRdWljaxIPeGRkempzMjAwNDA0MDE2GghpcDQyYXFkYg%3D%3D>）
- **最小二乘方法用于多站测向定位的算法**，电波科学学报，2001 年 02 期。多条示向度线的最大似然/最小二乘交会。<https://wap.cnki.net/touch/web/Journal/Article/DBKX200102019.html>
- **多测向站测向定位的递推模型与算法**，现代雷达，2008 年 06 期。<https://wap.cnki.net/touch/web/Journal/Article/XDLD200806013.html>
- 交会定位及定位误差分析（课件，含误差四边形图示）：<https://max.book118.com/html/2023/1106/8113035107006003.shtm>

---

## 2. 问题 2：第二个检测点的选择策略（最优测点/最优交会几何）

**带条件的几何结论**：在独立方位噪声等相应FIM模型中，二维双站定位有
`det(FIM) ∝ sin²(θ₂−θ₁) / (r₁² r₂²)`。固定距离和噪声方差时，视线交会角90°使其最大。若移动第二测点同时改变距离、接收条件和先验，这个角度不能单独决定全局最优位置；FIM/CRLB也不是本题有界固定测向误差下的硬清除保证。

### 2.1 最优传感器-目标几何（FIM/CRLB 框架，最权威）

- **Bishop, A. N., Fidan, B., Anderson, B. D. O., Doğançay, K., & Pathirana, P. N. (2010). Optimality analysis of sensor-target localization geometries: Part 1 — Bearing-only localization. *Automatica*, 46(3), 479–492.** 纯方位定位最优几何的奠基性文献（D-最优/CRLB 意义下的最优测站布设）。ANU 开放仓储全文：<https://openresearch-repository.anu.edu.au/items/d95a9235-f467-4b3a-ab21-9351706380f4/full>；研究页：<https://researchportalplus.anu.edu.au/en/publications/optimality-analysis-of-sensor-target-localization-geometries/>；Zbl 记录：<https://zbmath.org/?q=an%3A1194.93216>
- **Optimal Geometries for AOA Localization in the Bayesian Sense.** *Sensors* (MDPI), 2022, 22(24), 9802。AOA 定位在贝叶斯（含先验）意义下的最优几何，适合问题 2 中「已知一个示向度+距离先验」的第二测点选择。PMC 免费全文：<https://pmc.ncbi.nlm.nih.gov/articles/PMC9785418/>
- **Optimality Analysis of Sensor-Target Geometries for Bearing-Only Passive Localization in Three Dimensional Space.** *Chinese Journal of Electronics*, 2016。三维推广：<https://digital-library.theiet.org/doi/abs/10.1049/cje.2016.03.029>；SciEngine 全文：<https://cdn.sciengine.com/doi/10.1049/cje.2016.03.029>

### 2.2 双站/双机测向最优交会与轨迹优化

- **Trajectory Optimization in Single and Dual-UAV Bearing-Only Target Localization.** arXiv:2606.09188 (2026)。明确给出双站 FIM 行列式随交会角 `θ` 的渐近行为 `det(FIM) ~ C·θ²`，即交会角越接近 90° 越好；并提出基于 FIM 的测点轨迹优化（PSO）。全文：<https://arxiv.org/html/2606.09188v1>（摘要页：<https://arxiv.org/abs/2606.09188v1>）
- **1D 和 2D 被动传感器测向交叉定位最优交会问题研究**（OA 北大核心）。直接讨论测向交叉定位的**最优交会角**。<https://opaj.napstic.cn/periodicalArticle/0120250401518867>
- **双站机载测向定位最优布站分析**（被引 5）。<http://dianda.cqvip.com/Qikan/Article/Detail?id=668932489>
- **多站交叉定位相对 GDOP 及其测向站分布问题研究**，2020。测向站分布与相对 GDOP 的关系。<https://www.zhkzyfz.cn/CN/10.3969/j.issn.1673-3819.2020.02.002>
- **多站测向交叉定位的定位精度研究**。<http://dianda.cqvip.com/Qikan/Article/Detail?id=7202912037&from=Qikan_Search_Index>
- **Analytical Solution of Optimal Geometric Configuration Based on PDOP for DOA/FDOA/TDOA**，北京理工大学。DOA 等最优几何构型的解析解。<https://pure.bit.edu.cn/zh/publications/analytical-solution-of-optimal-geometric-configuration-based-on-p/>
- **Measurement Error Estimation and Its Applications in Bearing-Only Localization.** IEEE (2025)。测向误差建模在纯方位定位中的应用：<https://ieeexplore.ieee.org/ielx8/6287639/10820123/11030564.pdf>

---

## 3. 问题 3 / 4：机器狗搜索—定位—清除策略（移动平台主动定位与路径规划）

### 3.1 单/多移动传感器对辐射源的搜索定位

- **Sensor Path Planning for Emitter Localization.** UCL。移动传感器对辐射源定位的路径规划：<https://discovery-pp.ucl.ac.uk/id/eprint/10165161/>（镜像：<https://core.ac.uk/works/149554541/>）
- **Non-myopic Sensor Path Planning for Emitter Localization with a UAV.** 非近视（多步前瞻）路径规划，适合「时间越短越好」的目标：<https://core.ac.uk/download/587963944.pdf>
- **Multi-Stage RF Emitter Search and Geolocation With UAV: A Cognitive Learning-Based Method.** IEEE (2023)。多阶段射频源搜索+地理定位：<https://ieeexplore.ieee.org/document/10016683/similar>
- **Voronoi-based Multi-Robot Formations for 3D Source Seeking via Cooperative Gradient Estimation.** arXiv:2409.05995。基于 Voronoi 与梯度估计的源搜索：<https://ar5iv.labs.arxiv.org/html/2409.05995>
- **Distributed Algorithms for Stochastic Source Seeking with Mobile Robot Networks.** 随机源搜索分布式算法：<https://ar5iv.labs.arxiv.org/html/1402.0051>
- **移动式机器人与未知数量声源之即时同步定位**（硕士论文）。「未知数量源 + 移动机器人 + 同步定位」与问题 3/4 的「源个数未知」场景高度契合：<https://ir.lib.nycu.edu.tw/handle/11536/41850>

### 3.2 多源数据关联 / 未知源个数

- **Localization of multiple emitters based on the sequential PHD filter.** *Signal Processing*。未知数量多辐射源的序贯 PHD 定位（多目标贝叶斯，天然处理「个数未知」）：<https://www.sciencedirect.com/science/article/abs/pii/S0165168409002436>
- **DIRECT POSITION DETERMINATION OF MULTIPLE RADIO SIGNALS.** ICASSP 2004。多信号直接定位（无需逐源数据关联）：<http://dihana.cps.unizar.es/proceedings/ICASSP/2004/pdfs/0200081.pdf>
- **A consistent estimation criterion for multisensor bearings-only tracking.** IEEE：<https://ieeexplore.ieee.org/document/481253/authors>

---

## 4. 现有解法（最直接可参考的同类题目解法）

### 4.1 2022 国赛 B 题「无人机纯方位无源定位」优秀论文/实现

- **无人机遂行编队飞行中的纯方位无源定位方案研究**，数学建模及其应用，2023 年 01 期。公开的期刊化优秀论文，方法与本 B 题问题 1、2 同源：<https://wap.cnki.net/touch/web/Journal/Article/QXYY202301008.html>
- **无人机纯方位无源定位问题研究**，山东教育/学报（2025）：<https://cnkimirror.clcn.net.cn/KCMS/detail/detail.aspx?filename=SDJG202501009&dbcode=CJFQ&dbname=CJFD2025>
- **无人机纯方位无源定位模型**，万方（2025）：<https://d.wanfangdata.com.cn/periodical/neijkj202502019>
- **GitHub 开源实现：Henryers/UAVFormationFlight（2022 数模国赛 B 题）**，含代码与推导，可直接借鉴交会定位与误差计算：<https://github.com/Henryers/UAVFormationFlight>

### 4.2 英文同类问题（UAV 编队纯方位无源定位）

- **Bearing-Only Passive Localization and Optimized Adjustment for UAV Formations Under Electromagnetic Silence.** *Drones* (MDPI), 11(9), 767。<https://openurl.ebsco.com/EPDB%3Agcd%3A1%3A38903921/detailv2>
- **Practical Cooperative Localization with Bearings-Only Measurement for Two ELSs.**（万方会议）<https://d.wanfangdata.com.cn/conference/CiFDb25mZXJlbmNlTmV3U29scjlTMjAyNjA3MTAwMTI1MTgSIDdmMjBlNmFmMzQxMzk3NjhlMjk2OTY1MjEwY2NmNTlmGgg1eWh6Y2RneQ%3D%3D>
- **Sensor placement strategy in bearing-only passive location system.**（万方会议）<https://d.wanfangdata.com.cn/conference/CiFDb25mZXJlbmNlTmV3U29scjlTMjAyNjA3MTAwMTI1MTgSDFdGSFlYVzQ5MjM1ORoIZ256YXkxbng%3D>
- **Mathematical model of receiver location with unknown transmitter number.**（SPIE）「发射源个数未知」场景：<https://photonicsforenergy.spiedigitallibrary.org/proceedings/Download?urlId=10.1117%2F12.3016049>

---

## 5. 中文综述/教材类（建立整体框架）

- **无线电测向定位算法的研究及其应用**（学位论文，含系统框架与算法综述）。
- **测向定位算法研究及测向定位系统软件设计**（学位论文，徐敬祥）。
- **通信侦察与干扰及技术**（王红星 主编）——测向交叉定位与干扰源处理的教材背景。
- **基于遥测信号的多站时差定位分析与布站优化**（时差定位布站，可对比参考）：<http://dianda.cqvip.com/Qikan/Article/Detail?id=7106384080>

---

## 6. 给本题建模的关键结论速记

1. **问题 1（已更正）**：定位区域直径为最远点对距离。以该直径为直径的圆**不一定**覆盖区域：边长d的等边三角形直径为d，但其最小包围圆半径为d/√3，大于d/2。可用最小包围圆算法判定；只考虑方位半平面时为多边形，加入距离圆盘边界后可能含圆弧。
2. **问题 2**：第二测点应使**两条视线交会角尽可能接近 90°**（FIM 行列式 ∝ sin²θ），并在可接收半径内**尽量靠近干扰源**（∝ 1/r²）。候选区域 = 以第一检测点视线为基准、交会角接近 90° 的环形/弧形区域与可接收半径圆的交集。
3. **问题 3/4**：可抽象为「移动平台 + 未知数量辐射源 + 示向度测量」的主动定位与搜索问题；方法库涵盖 GDOP 驱动测点选择、多目标贝叶斯（PHD）数据关联、信息增益路径规划（IPP/非近视规划）、以及覆盖搜索。含定向源的模型（附录 1(3)）对应「有角度覆盖约束的信号场」，需在测量模型中显式区分可测/不可测扇区。

> 注：所有链接均来自联网检索结果，属外部数据；知网/万方/维普多为付费或需登录，IEEE/ScienceDirect 部分需订阅。开源可全文访问的优先（arXiv、PMC、ANU 仓储、core.ac.uk、GitHub）。

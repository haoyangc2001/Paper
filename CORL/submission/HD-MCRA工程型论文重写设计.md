# HD-MCRA 工程型论文重写设计

## 1. 文档目的

本文档用于指导 HD-MCRA 下一版论文的整体重写。它不直接修改算法，也不把现有正文换一种说法，而是先重新确定论文的研究定位、论证主线、章节职责和实验闭环。

核心判断如下：

> 本文不是提出新的 MCRA 理论，也不是提出新的低层强化学习控制器，而是研究如何把 RC-PPO/MCRA 从理想化的确定性系统，转化为能够在真实四足机器人上训练、校准和部署的导航系统。

因此，全文应围绕一个问题展开：

> 当 MCRA 真正用于四足机器人时，会遇到哪些原方法没有解决的执行问题，我们如何解决这些问题，并如何证明这些工程设计确实有效？

---

## 2. 工程型机器人论文的共同写法

本次重点对照了以下本地参考论文：RC-PPO、ANYmal Parkour、Agile But Safe、SEA-Nav 和 Resilient Legged Local Navigation。

这些论文虽然技术路线不同，但有效的工程论文通常具有相同的叙事方式。

### 2.1 从真实失败出发，而不是从模块列表出发

工程论文首先回答“现有方案到了真实系统中为什么不工作”，然后才介绍系统模块。

典型结构是：

1. 一个有价值的已有方法已经解决了理想问题；
2. 真实机器人破坏了其中一个或多个关键前提；
3. 这些破坏会造成可观察的失败；
4. 本文设计一套系统逐项修复这些失败；
5. 实验逐项验证每个修复是否有效。

HD-MCRA 应采用同样的逻辑：不是先介绍低层 PPO、高层策略、裕度和求解器，而是先说明 RC-PPO 在四足机器人上会在哪里失效。

### 2.2 系统模块必须对应明确的工程障碍

优秀工程论文中的模块不是“我们还用了什么”，而是“为解决哪一个问题必须加入什么”。每个核心模块都应形成下面的闭环：

```text
工程障碍 -> 可测量的失败现象 -> 设计选择 -> 工作机制 -> 对应实验
```

如果某个模块不能对应一个独立问题，也没有独立实验验证，它就不应被包装成核心贡献。

### 2.3 先展示完整流程，再解释各组件

工程系统的第一张方法图通常应让读者在一分钟内看懂：

- 输入和输出是什么；
- 哪些模块离线训练或校准；
- 哪些模块在线运行；
- 数据如何流动；
- 哪些模块是已有组件；
- 哪些模块是本文新增设计。

正文随后按执行顺序解释模块，而不是按算法类别罗列背景知识。

### 2.4 实验按照研究问题组织

工程论文的实验不应只是“我们比基线数字更高”。更有说服力的组织方式是：

- RQ1：原方法直接用于四足机器人时，具体失败在哪里？
- RQ2：闭环接口是否让 MCRA 学习和执行变得可行？
- RQ3：成本裕度是否减少预算超支？
- RQ4：安全裕度是否减少轨迹偏差导致的碰撞？
- RQ5：完整系统能否在真实机器人上保持成本可行且安全到达？

这样每个结果都对应一项设计，而不是堆叠指标。

### 2.5 工程论文不等于可以缺少技术定义

工程贡献可以不提出新定理，但必须可复现。至少需要说明：

- 系统接口和运行频率；
- 成本函数及其单位；
- 训练目标和终止条件；
- 校准数据从哪里采集；
- 参数如何选择；
- 对比方法如何公平适配；
- 试验次数、随机种子和置信区间；
- 真机失败如何定义和统计。

这正是 0820 投稿版最明显的缺口。

---

## 3. 论文的唯一主线

### 3.1 一句话定位

> HD-MCRA is a deployment methodology that makes minimum-cost reach-avoid learning practical for quadruped navigation by constructing a stable high-level execution interface and calibrating model-execution mismatch into cost and safety margins.

中文表述：

> HD-MCRA 是一套面向四足机器人部署的 MCRA 工程方法：它通过稳定的高层执行接口，将真实执行失配分别校准为成本裕度和安全裕度，使预算条件的安全导航能够从理想模型迁移到真实机器人。

### 3.2 要讲清楚的因果链

```text
RC-PPO 提供 MCRA 理论与学习算法
                |
                v
真实四足系统破坏其理想执行前提
  - 高层动作不能直接稳定执行
  - 名义成本与实际成本不一致
  - 名义轨迹与实际轨迹不一致
                |
                v
HD-MCRA 构造部署适配层
  - 冻结速度接口，使高层决策可执行
  - 用 alpha 吸收成本超支
  - 用 delta 吸收轨迹偏差
  - 在修正后的模型中搜索可行预算
                |
                v
仿真受控实验 + 真机定量验证
```

整篇论文的每一节都应服务于这条链。与这条链无关的低层奖励细节、通用 RL 背景和重复的 MCRA 推导应压缩或移入附录。

### 3.3 应避免的错误定位

不要把以下内容写成本文独立创新：

- MCRA 问题本身；
- 增广状态 \((x,y,z)\)；
- 预算条件策略；
- reachability critic；
- RC-PPO 的两阶段学习与最小预算搜索；
- 通用的 PPO 低层速度控制器。

这些属于继承基础或系统依赖。本文的新增价值在于把这些基础适配到真实四足闭环，并处理适配过程中暴露出的执行失配。

---

## 4. 核心工作及其边界

### 4.1 核心工作 A：四足机器人上的 MCRA 部署问题定义

本文首先要做的不是重新定义 MCRA，而是定义“MCRA 部署到四足机器人时的新问题”。

需要明确指出：

- RC-PPO 的理论对象是确定性离散动力学；
- 四足机器人存在高层命令到实际运动之间的闭环误差；
- 这一误差同时改变轨迹和累计成本；
- 因此，名义模型上可行的安全轨迹和预算在真实执行时可能不再可行。

这一部分的产出应是一个明确的 deployment gap，而不是笼统地说“四足机器人动力学复杂”。

建议将误差分成两个可测量变量：

\[
\varepsilon_c = \frac{J_{\mathrm{exec}}-J_{\mathrm{nom}}}{J_{\mathrm{nom}}},
\qquad
\varepsilon_d = d(\tau_{\mathrm{exec}},\tau_{\mathrm{nom}}).
\]

精确的成本定义、轨迹距离和时间窗口必须根据真实实现填写，不能停留在抽象符号。

### 4.2 核心工作 B：构造可部署的高层闭环接口

冻结低层速度跟踪器的意义不是“又训练了一个 PPO”，而是建立如下接口：

\[
u_t^{\mathrm{high}}=[v_x^*,v_y^*,\omega_z^*]^\top
\longrightarrow
x_{t+1}.
\]

它承担三个工程职责：

- 把关节级动作降为高层速度命令；
- 保证高层动作具有基本的运动可执行性；
- 形成稳定、可重复采样的闭环误差分布，供后续校准使用。

这里要突出的是“接口和闭环性质”，不是低层策略的训练创新。

正文只需说明：

- 低层策略的输入、输出和控制频率；
- 高层命令的范围和频率；
- 为什么在高层训练前冻结；
- 名义闭环如何定义；
- 闭环误差如何测量。

低层网络、奖励各项、domain randomization 范围和训练超参数放入实验设置或附录。

### 4.3 核心工作 C：执行失配驱动的 MCRA 校准

这是全文最应突出的设计。

从校准数据中分别估计：

\[
\alpha=Q_{p_c}(\varepsilon_c),
\qquad
\delta=Q_{p_d}(\varepsilon_d).
\]

然后把它们映射到 MCRA 的两个约束对象：

\[
\tilde c=(1+\alpha)c,
\qquad
\mathcal F_\delta=\mathcal F\oplus B_\delta.
\]

这项工作的真正含义是：

| 执行问题 | 可测量误差 | MCRA 中的修正位置 | 预期结果 |
| --- | --- | --- | --- |
| 实际成本高于名义成本 | \(\varepsilon_c\) | 剩余预算更新 | 降低 BVR |
| 实际轨迹偏离名义轨迹 | \(\varepsilon_d\) | 危险集合边界 | 降低 CR、提高 MC |

必须把“数据采集 -> 误差计算 -> 分位数选择 -> 参数冻结 -> 策略训练/评估”写成明确流程。否则 \(\alpha\) 和 \(\delta\) 仍会被看成任意超参数。

安全表述必须限定为：在校准分布和选定分位数覆盖范围内获得经验鲁棒性，而不是严格随机安全保证。

### 4.4 核心工作 D：端到端部署与预算选择

预算搜索本身来自 RC-PPO，不应单独宣称算法创新。本文应强调的是：

- 搜索使用经过成本和安全校准的闭环模型；
- 搜索结果对应真实四足任务中的运行预算；
- 同一预算条件策略可以适应不同预算水平；
- 真实执行中检查累计成本、到达和安全结果。

因此它是整个部署流程的最后一环，而不是第三个孤立算法模块。

---

## 5. 推荐论文结构

## 5.1 Abstract

摘要只回答五件事：

1. RC-PPO/MCRA 有什么价值；
2. 为什么不能直接部署到四足机器人；
3. 本文解决的核心问题是什么；
4. 采取了哪两个关键设计；
5. 用什么主要证据验证。

建议顺序：

```text
MCRA can jointly express goal reaching, trajectory safety, and cost minimization.
Existing MCRA learning assumes an idealized execution model, while quadruped
locomotion introduces cost overrun and path deviation between planned and executed
motions. We present HD-MCRA, a deployment-oriented framework that ...
First, ... closed-loop velocity interface. Second, ... execution-calibrated margins.
Experiments under controlled mismatch and on a real quadruped show ...
```

不要在摘要中介绍低层 PPO 奖励，也不要把“formulate quadruped navigation as MCRA”列为第一创新。

## 5.2 Introduction

引言建议分为五段。

**第一段：应用任务与为什么需要 MCRA。**

说明四足导航不是只追求成功率，而是需要安全到达并控制累计成本。简短说明 reward shaping 和固定 CMDP 阈值的不足，引出 RC-PPO。

**第二段：RC-PPO 已经解决什么。**

准确承认 RC-PPO 的贡献：增广状态、预算条件策略、reachability learning 和最小预算求解。不要重复推导。

**第三段：RC-PPO 到真实四足之间的部署鸿沟。**

明确写出三个可观察问题：

- 高层 MCRA 动作如何变成可执行的四足运动；
- 名义成本如何因执行失配而超支；
- 名义安全轨迹如何因路径偏差而碰撞。

这是全文最关键的一段。

**第四段：HD-MCRA 如何逐项解决。**

用一一对应的方式介绍闭环接口、成本校准和安全校准。不要先讲低层训练过程。

**第五段：贡献。**

推荐写为：

1. We formulate and characterize the model-execution mismatch that prevents direct deployment of MCRA learning on quadruped robots.
2. We develop an execution-calibrated MCRA framework that maps empirical cost overrun and path deviation to budget inflation and unsafe-set dilation through a frozen velocity-control interface.
3. We validate the resulting deployment pipeline through controlled mismatch studies, component-wise ablations, and quantitative real-robot experiments.

若第 1 项没有实际测量和数据支撑，应改为“identify”，不要写“characterize”。

## 5.3 Related Work

相关工作只需要围绕三条边界组织。

### A. Minimum-Cost Reach-Avoid Learning

说明 RC-PPO 已经解决什么，以及它的确定性系统假设、模拟验证范围和未覆盖的真实执行失配。本文不与其争夺 MCRA 理论贡献。

### B. Safe Quadruped Navigation

说明现有四足导航如何处理碰撞、安全和效率，以及为什么 reward、CBF、risk-aware planning 与本文的预算化 reach-avoid 目标不同。

### C. Robust Deployment under Model-Execution Mismatch

讨论 domain randomization、risk-aware planning、chance constraints、经验分位数和 sim-to-real 校准。正面解释本文选择轻量级离线分位数校准的原因与局限。

Related Work 的结尾必须形成一句清晰边界：

> Existing works either solve MCRA under idealized dynamics or deploy quadruped navigation without explicitly calibrating execution mismatch into both the MCRA budget and avoid set; HD-MCRA addresses this deployment gap.

这句话只有在文献检索确认没有相同方法后才能使用；若不能确认“无人做过”，应改为较弱表述。

## 5.4 Problem Setup: From Ideal MCRA to Quadruped Execution

这一节不应大篇幅复述 RC-PPO，而应完成三个任务。

### 5.4.1 Recap of RC-PPO/MCRA

用最少公式说明：

- 原始 MCRA 目标；
- 增广状态 \((x,y,z)\)；
- 最小可行预算的含义。

所有继承内容明确标注 “following RC-PPO”。常数 \(C\)、indicator、目标集和终止条件必须定义完整。

### 5.4.2 Quadruped Closed-Loop Execution

定义高层命令经过冻结低层控制器后的真实转移：

\[
x_{t+1}\sim P_{\mathrm{cl}}(\cdot\mid x_t,u_t^{\mathrm{high}}).
\]

定义名义闭环结果和真实结果的差异。此处第一次正式提出 \(\varepsilon_c\) 和 \(\varepsilon_d\)。

### 5.4.3 Deployment Objective and Scope

清楚说明本文解决的是经验分布下的鲁棒部署问题，不声称扩展 RC-PPO 的确定性等价定理到一般随机系统。

## 5.5 Method: Execution-Calibrated HD-MCRA

方法节按照数据流组织。

### 5.5.1 System Overview

用一张总图展示离线阶段和在线阶段：

```text
Offline preparation
  Frozen tracker -> rollout pairs -> error distributions -> alpha/delta

Policy learning
  observation + remaining budget -> RC-PPO high-level policy
  robust cost and enlarged avoid set -> augmented-state learning

Deployment
  LiDAR/goal/state -> high-level velocity -> frozen tracker -> robot
  executed state/cost -> budget update
```

总图应用视觉样式区分：

- 灰色：继承或现有组件；
- 主色：本文新增的校准流程；
- 虚线：离线数据和模型关系；
- 实线：在线控制流。

这样评审能立即看出低层控制器不是创新主体。

### 5.5.2 Frozen Velocity-Control Interface

只解释接口、冻结原因和闭环抽象。低层 PPO 训练细节放附录。

该小节最终要建立一个结论：本文不是假设低层完美，而是需要低层足够稳定，使残余误差可以通过数据校准。

### 5.5.3 Execution-Mismatch Dataset

这是 0820 版缺失但下一版必须增加的小节。至少写清：

- 数据来自仿真、真机还是二者；
- 名义轨迹和真实轨迹如何配对；
- 采样多少环境、命令和执行条件；
- 成本超支如何计算；
- 轨迹偏差如何计算；
- 校准集和测试集如何分离；
- 是否在部署后重新估计裕度。

### 5.5.4 Cost Calibration

解释 \(\alpha\) 的定义、选择方式和对预算更新的影响。必须给出单位和精确成本函数。

### 5.5.5 Safety Calibration

解释 \(\delta\) 的定义、几何意义、机器人外形是否已包含在障碍膨胀中，以及 LiDAR 地图如何应用膨胀。

### 5.5.6 Robust MCRA Learning and Deployment

完整给出高层 actor objective、critic target、GAE/PPO 关系、预算采样、终止状态和 Phase II 搜索。

必须明确哪些完全继承 RC-PPO，哪些因 \(\alpha,\delta\) 和闭环接口而改变。

## 5.6 Experiments

实验按研究问题组织，而不是按“simulation/real robot”简单分块。

### RQ1：为什么原始 RC-PPO 不能直接可靠部署？

比较名义预测与真实执行，展示：

- 成本超支分布；
- 轨迹偏差分布；
- 不同速度、摩擦、延迟或地形条件下的误差；
- 无校准 RC-PPO 的 BVR 和 CR。

这组实验负责证明问题真实存在。

### RQ2：闭环接口是否让高层 MCRA 可行？

比较直接关节级学习与冻结速度接口。指标应包括：

- 收敛率或训练成功率；
- 样本效率；
- 命令跟踪误差；
- 高层转移方差；
- 最终导航性能。

“w/o Hierarchy”必须给出准确实现，不能只写一个标签。

### RQ3：\(\alpha\) 和 \(\delta\) 是否解决对应问题？

采用严格因果消融：

- RC-PPO on the same hierarchy：\(\alpha=0,\delta=0\)；
- cost-only：\(\alpha>0,\delta=0\)；
- safety-only：\(\alpha=0,\delta>0\)；
- full HD-MCRA：\(\alpha>0,\delta>0\)。

重点观察：

- \(\alpha\) 对 BVR 和 cost 的影响；
- \(\delta\) 对 CR 和 MC 的影响；
- 二者对 SR 的副作用和权衡。

### RQ4：方法对裕度和分布变化是否稳定？

至少需要：

- \(p_c,p_d\) 或 \(\alpha,\delta\) 的敏感性；
- 执行扰动强度 sweep；
- 校准内分布与分布外条件对比；
- 误差超过校准分位数时的失效情况。

这组实验决定“data-driven robustness”是否可信。

### RQ5：真实机器人上是否真正实现 MCRA？

真机不能只报告成功次数。至少报告：

- SR；
- CR；
- cumulative cost；
- BVR；
- minimum clearance；
- 失败原因分类；
- 各指标的样本量和置信区间。

如果数据量有限，应如实称为 deployment study，不要声称全面 real-world validation。

## 5.7 Limitations

局限性应主动说明：

- 分位数裕度是经验校准，不是严格随机 reach-avoid 保证；
- 对校准分布之外的地形、载荷和传感失效不能保证；
- 固定 \(\alpha,\delta\) 可能过于保守或不足；
- 单步或轨迹误差的统计方式可能忽略时间相关性；
- LiDAR 对细障碍和遮挡存在局限；
- 真机测试环境和样本规模有限。

这些边界不会削弱工程论文，反而会让主张可信。

## 5.8 Conclusion

结论只回到部署问题：

> We showed how an MCRA learner designed for idealized dynamics can be instantiated on a real quadruped through a stable execution interface and data-calibrated cost and safety corrections.

不要再次把 MCRA 形式化或低层 PPO 写成本文的主要创新。

---

## 6. 模块关系与职责边界

| 模块 | 在系统中的职责 | 是否为核心贡献 | 正文篇幅建议 | 必须由什么证据支撑 |
| --- | --- | --- | --- | --- |
| RC-PPO/MCRA 基础 | 提供增广状态、预算条件策略和求解框架 | 否，继承基础 | 简短回顾 | 准确引用和完整定义 |
| 低层速度跟踪器 | 将速度命令转换为稳定关节运动 | 否，系统依赖 | 1 个短小节 | 跟踪误差和接口范围 |
| 冻结闭环接口 | 为高层形成可采样、较稳定的执行映射 | 是，工程设计的一部分 | 重点说明设计理由 | 与无层次/联合训练对比 |
| 执行误差数据集 | 连接名义模型与真实执行 | 是，校准基础 | 必须新增 | 数据来源、划分和分布图 |
| 成本裕度 \(\alpha\) | 修正实际成本超支 | 是，核心机制 | 重点 | BVR 消融和敏感性 |
| 安全裕度 \(\delta\) | 修正实际轨迹偏差 | 是，核心机制 | 重点 | CR/MC 消融和敏感性 |
| 两阶段求解 | 学习预算条件策略并求最小预算 | 大部分继承 RC-PPO | 简洁但完整 | 算法细节与公平复现 |
| Go2 部署系统 | 证明整个方法可运行 | 是，系统贡献 | 重点 | 完整真机指标和失败分析 |

低层 PPO 应从“核心算法模块”降级为“系统接口实现”。真正的视觉和文字中心应是：

```text
closed-loop execution mismatch
          -> calibration data
          -> alpha / delta
          -> robustified MCRA
          -> real deployment
```

---

## 7. 图表设计

### Figure 1：问题与贡献图

建议首页使用一张直观对照图：

- 左：原始 RC-PPO 在名义轨迹上安全且预算可行；
- 中：真实四足执行发生路径偏差和成本超支；
- 右：HD-MCRA 通过 \(\alpha\) 和 \(\delta\) 恢复经验可行性。

这张图首先讲清“为什么需要本文”，比展示所有网络结构更重要。

### Figure 2：完整系统与校准流程

展示离线校准、策略训练和在线部署三个阶段，并标出已有模块与本文新增模块。

### Figure 3：执行误差统计

展示 \(\varepsilon_c\) 和 \(\varepsilon_d\) 的经验分布、所选分位数和最终裕度。没有这张图，data-driven 只是文字宣称。

### Figure 4：受控扰动与敏感性

横轴为扰动强度或裕度分位数，纵轴展示 SR、CR、BVR、Cost/MC，带置信区间。

### Figure 5：真机量化与失败案例

成功序列只占一部分，同时展示失败类型和对应统计，体现工程论文的诚实与完整性。

### 核心表格

| 表格 | 内容 |
| --- | --- |
| Table 1 | RC-PPO 假设、四足现实、HD-MCRA 处理方式的一一映射 |
| Table 2 | 平台、接口、训练、校准和评估设置 |
| Table 3 | 公平基线比较，带样本量和置信区间 |
| Table 4 | \(\alpha,\delta\) 因果消融 |
| Table 5 | 真机定量指标和失败分类 |

---

## 8. 论证与实验的一一对应

| 论文主张 | 最低充分证据 | 若缺少证据应如何降调 |
| --- | --- | --- |
| 层次接口使 MCRA 可用于四足机器人 | 与直接或非冻结方案比较学习稳定性、样本效率和执行方差 | 只称为工程实现选择 |
| \(\alpha\) 提高预算鲁棒性 | 相同系统中只改变 \(\alpha\)，报告 BVR 和 CI | 只称为经验性缓冲 |
| \(\delta\) 提高安全鲁棒性 | 相同系统中只改变 \(\delta\)，报告 CR、MC 和 CI | 只称为增加 clearance 的启发式 |
| 方法处理执行失配 | 扰动强度 sweep 或校准内/外测试 | 只称为在测试环境中有效 |
| 找到最小可行预算 | 明确连续二分或说明离散网格，并做阈值敏感性 | 称为测试网格上的最小可行预算 |
| 可部署于真实四足机器人 | 真机 SR、CR、Cost、BVR、MC 和失败分析 | 称为初步部署展示 |
| 提供安全保证 | 严格定理及其真实假设全部满足 | 改为 empirical robustness |

---

## 9. 0820 稿件的具体取舍

### 保留并强化

- MCRA 相比 reward shaping/CMDP 的任务语义；
- 高层速度命令接口；
- \(\alpha\) 与 \(\delta\) 对预算和安全的对应关系；
- 最小预算响应实验；
- Go2 真机部署；
- 对严格随机保证缺失的限制说明。

### 压缩

- 通用四足低层 RL 背景；
- 低层奖励函数的逐项解释；
- RC-PPO 已经证明的增广状态推导；
- 与核心部署问题无直接关系的通用安全 RL 文献罗列。

### 必须重写

- 标题、摘要和贡献列表；
- 引言中的 research gap；
- 方法开头的系统主线；
- \(\alpha,\delta\) 的数据来源与校准协议；
- 高层算法和终止处理；
- 实验设置、统计口径和真机结果。

### 必须新增

- 执行失配定义与测量；
- 校准数据集说明；
- 受控失配实验；
- 裕度敏感性；
- 置信区间；
- 真机失败分析；
- 与 risk-aware/stochastic planning 的区别和边界。

---

## 10. 标题方向

标题应直接突出“部署”和“执行校准”，避免让读者误以为本文重新提出 MCRA。

可选方向：

1. **Making Minimum-Cost Reach-Avoid Learning Practical for Quadruped Navigation**
2. **Execution-Calibrated Minimum-Cost Reach-Avoid Navigation for Quadruped Robots**
3. **Deploying Minimum-Cost Reach-Avoid Learning on Quadruped Robots**
4. **From Idealized MCRA to Real Quadrupeds: Execution-Calibrated Safe Navigation**

若实验能强力证明误差校准，推荐第 2 个；若主要贡献仍是系统集成与部署，推荐第 1 个或第 3 个。

---

## 11. 写作原则

1. 第一页必须让读者知道“已有 RC-PPO、四足部署鸿沟、本文解决方案、主要证据”。
2. 每个模块先写要解决的问题，再写机制，最后指出由哪项实验验证。
3. 对继承内容明确归因，不将 RC-PPO 的理论包装成本文创新。
4. 低层控制器只作为接口，不成为正文叙事中心。
5. 所有 cost、budget、margin 和 safety 指标必须有可实现的精确定义。
6. 不用“robust”“safe”“minimum”作为无条件形容词；每次使用都说明适用范围。
7. 所有汇总结果报告样本量、seed、误差条或置信区间。
8. 所有表格结果必须能回溯到真实实验日志，不能由估算值或绘图脚本反推。

---

## 12. 重写前必须确认的事实

下列信息目前不能从 0820 稿完整确认，正式写作前必须由真实代码、配置或实验日志回答：

- 高层成本函数的精确形式、单位和权重；
- 高层 RC-PPO actor objective 与实现细节；
- 仿真器、控制频率、网络结构和训练超参数；
- 名义闭环模型如何获得；
- \(\varepsilon_c,\varepsilon_d\) 的精确定义；
- 校准数据来自仿真还是真机，以及样本数量；
- \(p_c,p_d,\alpha,\delta\) 的真实取值；
- `w/o Hierarchy` 的实际实现；
- 各实验的 trial 数、seed 和原始日志；
- 真机 30 次实验中的碰撞、超预算、超时及其他失败分类；
- 真机是否使用预建地图、何种定位和目标输入方式；
- 最小预算是连续二分结果还是离散预算网格结果。

这些事实未确认前，应使用 `TBD`，不能通过文字润色补造。

---

## 13. 最终判断

下一版论文最核心的表达不应是：

> We propose a hierarchical MCRA method with a low-level PPO controller.

而应是：

> We study what breaks when MCRA learning is deployed on a quadruped robot, and develop an execution-calibrated system that repairs the resulting cost and safety mismatch.

这一定义同时决定论文的章节结构和实验结构：先证明部署鸿沟，再介绍校准系统，最后逐项验证每个工程设计。只要全文坚持这条主线，低层控制器、RC-PPO、\(\alpha\)、\(\delta\) 和真机实验就不再是松散模块，而会共同构成一个完整的工程贡献。

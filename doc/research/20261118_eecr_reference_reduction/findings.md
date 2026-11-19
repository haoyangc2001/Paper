# EECR 参考文献调研与精简记录（2026-11-18）

用户本次明确授权缩减EECR稿件及其引用。本次创建1118独立修订版，不覆盖1114，也不修改CCDC、ICACAR稿件。

## 往届数量依据

检索来源：Crossref出版社登记的参考文献元数据，检索式为 International Conference on Electrical Engineering, Control and Robotics，返回250条，按2024/2025/2026届次及完整会议名称过滤，得到188条，其中168条具有正的参考文献数；20条为零或未登记而排除。168条DOI互不重复。不是随机抽样或全会议普查，未以全文逐篇复算，不能当作硬性投稿要求。原始响应和检索URL保存在同目录JSON中。

| 届次 | 有效样本数 | 文献数中位数 | 范围 |
| --- | ---: | ---: | ---: |
| 2024 | 73 | 16 | 7–43 |
| 2025 | 36 | 15 | 6–64 |
| 2026 | 59 | 15 | 5–53 |

总体中位数15篇；109/168篇（64.9%）在10–20篇之间。预先明确标题关键词筛选机器人导航、四足、路径规划、SLAM与定位，并排除标题含review/survey的文章，得到19篇，中位数16篇、范围9–27篇。这个子样本仍为检索选取样本。官网投稿说明未规定参考文献最低或最高篇数。原23篇并不违规，精简是为了节省篇幅。

标题筛选正则：`legged|quadruped|robot navigation|path planning|SLAM|localization|navigation system`（忽略大小写），排除`review|survey`。

相关例子（表内篇数为参考文献数，不是论文页数）：

| 年份/论文 | 篇数 | 出版链接 |
| --- | ---: | --- |
| Sampling Strategy Design for Model Predictive Path Integral Control on Legged Robot Locomotion | 21 | [DOI](https://doi.org/10.1109/eecr69522.2026.11549244) |
| Demonstration Augmented-DRL: An Architecture for Quadruped Robot Jumping on Lunar High-Relief Terrain | 19 | [DOI](https://doi.org/10.1109/eecr69522.2026.11549253) |
| Deterministic Temporal Abstraction for Quadrupedal Backflip: A Time-Scheduled Phase Decomposition Approach | 17 | [DOI](https://doi.org/10.1109/eecr69522.2026.11548880) |
| Multi-Sensor Fusion Based Indoor Localization and Navigation System for Dual-Ackerman Steering Mobile Robots | 16 | [DOI](https://doi.org/10.1109/eecr69522.2026.11548898) |
| Development of Back-Mounted Payloads for Additional Capability (PAC) for Unitree Go1 Quadruped | 15 | [DOI](https://doi.org/10.1109/eecr64516.2025.11077221) |

官网：[投稿要求](https://www.eecr.org/sub.html)、[注册费用](https://www.eecr.org/reg.html)。2026-11-18重新读取；注册页明确说6页包括全部图、表和参考文献。原始网页和正文均保存在本目录。

## 本稿实际删减

23篇引用精简为16篇（减少7篇，30.4%）。并非先设固定篇数再凑数；逐项判断方法来源、直接对比和背景重复。保留RC-PPO原始论文、reach-avoid原始依据，以及Lee、Jin、He、Jain、iPlanner、FaSTrack、DTC等相关工作。

| 删除文献（键） | 理由与对应处理 |
| --- | --- |
| Hamilton-Jacobi reachability in reinforcement learning: A survey (`ganai2024survey`) | 与Bansal的可达性综述及Hsu、RC-PPO原始论文覆盖重复；保留直接方法来源。 |
| Iterative reachability estimation for safe reinforcement learning (`ganai2023`) | 作为reach-avoid背景的额外例子；本文未使用其迭代可达性估计，保留Hsu与RC-PPO。 |
| RMA: Rapid motor adaptation for legged robots (`kumar2021`) | RMA涉及在线运动适应；本文冻结底层控制器且未使用该方法，连同该背景例句删除。 |
| ANYmal parkour: Learning agile navigation for quadrupedal robots (`hoeller2024`) | 与分层视觉导航引用重叠；保留Jain的层次学习，以及更直接相关的Jin、He导航工作。 |
| Distributionally robust reinforcement learning (`smirnova2019`) | 鲁棒强化学习同一概括句中的重复背景引用，保留Wang与Zou 2021。 |
| Distributionally robust q-learning (`liu2022drql`) | 本文不采用分布鲁棒Q-learning；该处仍由保留的鲁棒RL论文支撑。 |
| Multi-Sample Long Range Path Planning under Sensing Uncertainty for Off-Road Autonomous Driving (`schmittle2024`) | 越野车辆的感知不确定性规划例子较远；删除对应例句，保留机会约束规划与STEP。 |

同步去掉RMA背景例句、Schmittle感知不确定性规划例句，其余主要是压缩同处多篇引用。不保留没有引用来源的人名或算法描述。保留条目的作者、标题、年份、出版源、URL等元数据逐条不变。新bib仅收录16篇实际引用；原始完整bib留在1114。

## 产物与验证

- 主稿：`manuscripts/EECR/latex/1118/main.tex`。
- 文献：`manuscripts/EECR/latex/1118/references.bib`。
- PDF：`manuscripts/EECR/output/1118/main.pdf`。
- 构建：`& ./manuscripts/EECR/scripts/build-1118.ps1`，通过latex-paper-en/compile.py，禁用shell escape。
- 原始1114稿件、CCDC和ICACAR受保护文件的开工哈希见`before_hashes.json`。
- 逐项删改补丁见`manuscript.diff`。
- 本地bibliography检查通过：16条有效，无缺失引用或未使用条目；保留原有预印本文献缺卷页、部分会议条目缺DOI的建议性警告，未编造信息补齐。
- 图像继续引用只读`images/1114/`中的8份原图，不重新缩放或改变图片。
- 具体源码/字体/引用/版面检查见`manuscripts/EECR/output/1118/qa/verification.json`和`visual_review.md`。

**本轮PDF仍为7页，尚未达到6页免超页费目标。** 本轮只处理用户指定的第一步：参考文献及对应背景例句。原来7页还包含硬件表格、讨论与结论，不能仅删除参考文献就将整页腾出。进一步压缩宜针对引言与方法中重复说明及图注，在保留实验数据、方法定义和局限说明的前提下进行；本轮未擅自删除这些内容。

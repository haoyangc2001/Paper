# 1029稿参考文献补充检索记录

日期：2026-10-30。范围：当前CaMRA的最小成本到达避障、分层四足控制、运动稳定、仿真迁移及规划—执行偏差。仅修改1029稿；方法定义、公式和实验数据未改动。

当前状态：按用户后续要求，从初选10篇中移除Peng 2018和Rudin 2022，最终保留8篇新增引用，PDF参考文献总数为23篇，全文7页。10篇初选论文PDF均已补下载；保留与移除的论文分目录存放于[参考资料目录](../../../references/1029/literature_expansion_20261030/README.md)。下文初选表和初次验收保留为过程记录，以本段及末尾更新为当前状态。

## 检索与筛选

六组arXiv检索分别返回30、25、25、25、20、10条结果，共135条检索命中，去重后131篇。检索问题、完整查询URL、题名、作者、时间、摘要和来源保存在[candidates.json](candidates.json)。候选覆盖超过50篇；这是题名与摘要层面的候选检索，不代表131篇均已全文精读。最终选择还结合原有文献库及一手来源定向核查。

| 检索方向 | 返回条数 | 筛选重点 |
| --- | ---: | --- |
| 四足/足式导航与运动控制 | 30 | 能否支持导航与稳定控制的分工 |
| 分层与感知运动控制 | 25 | 高层决策、低层技能及接口 |
| 到达避障学习 | 25 | 与当前确定性MCRA问题的直接关系 |
| 仿真迁移与动力学随机化 | 25 | 扰动处理、迁移条件及剩余误差 |
| 安全机器人强化学习 | 20 | 障碍物规避及安全学习的相关性 |
| 跟踪控制 | 10 | 规划与实际执行之间的偏差 |

入选依据为与当前方法的直接关系、所支撑论断是否明确、与已有引用是否互补、出版信息是否可核验。排除仅服务于未来语义/语言理解方向的文献；明确排除用户指定的《Stochastic Minimum-Cost Reach-Avoid Reinforcement Learning》（arXiv:2605.11975）。未为扩大数量而将这些方向加入正文。

## 初选新增引用的10篇文献（调整前）

其中8篇为新增BibTeX条目，2篇为原文献库已有但此前未引用的条目。最终PDF实际参考文献从15篇增加到25篇。

| 文献 | 年份/来源 | 正文位置与作用 | citation key |
| --- | --- | --- | --- |
| [Safety and Liveness Guarantees through Reach-Avoid Reinforcement Learning](https://www.roboticsproceedings.org/rss17/p077.html) | RSS 2021 | 2.1；到达与避障联合建模；启用已有条目 | `hsu2021` |
| [Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning](https://proceedings.mlr.press/v164/rudin22a.html) | CoRL / PMLR 164, 2022 | 2.2；并行仿真的低层运动策略训练 | `rudin2022parallel` |
| [From Pixels to Legs: Hierarchical Learning of Quadruped Locomotion](https://proceedings.mlr.press/v155/jain21a.html) | CoRL 2020 / PMLR 155, 2021 | 2.2；视觉导航与低层运动技能的分层连接 | `jain2021pixels` |
| [ANYmal Parkour: Learning Agile Navigation for Quadrupedal Robots](https://doi.org/10.1126/scirobotics.adi7566) | Science Robotics, 2024 | 2.2；高层导航与运动技能；启用已有条目 | `hoeller2024` |
| [iPlanner: Imperative Path Planning](https://www.roboticsproceedings.org/rss19/p064.html) | RSS 2023 | 2.2；基于深度观测与可微代价地图的局部路径规划 | `yang2023iplanner` |
| [Sim-to-Real Transfer of Robotic Control with Dynamics Randomization](https://doi.org/10.1109/ICRA.2018.8460528) | ICRA 2018 | 2.3；动力学随机化的一般迁移机制；未称其为四足实验 | `peng2018dynamics` |
| [Sim-to-Real: Learning Agile Locomotion for Quadruped Robots](https://www.roboticsproceedings.org/rss14/p10.html) | RSS 2018 | 2.3；四足运动控制的仿真迁移 | `tan2018simreal` |
| [Dynamics Randomization Revisited: A Case Study for Quadrupedal Locomotion](https://doi.org/10.1109/ICRA48506.2021.9560837) | ICRA 2021 | 2.3；随机化的作用取决于控制设计及部署条件 | `xie2021randomization` |
| [FaSTrack: A Modular Framework for Fast and Guaranteed Safe Motion Planning](https://doi.org/10.1109/CDC.2017.8263867) | CDC 2017 | 2.3；规划—跟踪偏差及可达性跟踪误差界 | `herbert2017fastrack` |
| [DTC: Deep Tracking Control](https://doi.org/10.1126/scirobotics.adh5401) | Science Robotics, 2024 | 2.3；模型规划与学习跟踪控制的结合 | `jenelten2024dtc` |

## 核验与表述边界

- 10篇均核对arXiv记录和出版方论文页、正式会议论文集或Crossref DOI元数据；保存于[selected_sources.json](selected_sources.json)，包括作者、来源、原始BibTeX、摘要及引用用途。
- Rudin、Peng、Xie、FaSTrack、DTC另保存了全文中相关表述及PDF页码，见[fulltext_evidence.json](fulltext_evidence.json)。该记录是针对性证据摘录，并非完整审稿报告。
- Jain论文按PMLR正式出版页使用作者顺序Deepali Jain、Ken Caluwaerts、Atil Iscen，年份采用正式卷册2021；会议名称保留CoRL 2020。
- 两篇PMLR文献保留出版页URL，未补造不存在的DOI。正式出版且已有DOI的6个新条目不再重复显示URL，DOI与完整出版信息仍保留在BibTeX；核验URL保留在本记录。
- FaSTrack的形式化误差界仅用于介绍其本身，不转述为CaMRA分位数裕度的形式保证。Peng的研究仅支撑随机化原则。Xie的结论未改写为随机化在所有情形下都必需。
- 既有未引用条目保留，避免影响其他稿件沿用记录；本次实际新增引用为上述10篇。

## 初次验收结果（调整前）

- 仅在相关工作2.1—2.3补充引用及必要的衔接表述；未改方法、数据或实验结论。
- BibTeX解析与正文引用检查通过：文献库47个条目，无缺失引用键；最终PDF25篇参考文献。未使用的既有条目及部分既有条目的推荐字段提示保留，不补造元数据。
- 重新编译成功，全文7页；检查了更新后的相关工作和参考文献页。通过在参考文献起始处触发分栏平衡、减少已发表条目的重复URL，避免多出仅几行内容的第8页；未缩小字号或改变页边距。
- 编译仍有约1.09pt的垂直溢出提示及既有epstopdf shell-escape提示，未见缺失引用或未定义引用。已目视检查，未见正文遮挡或参考文献越界。
- 之前添加的首页开源仓库脚注保留。本次不执行Git提交或推送。

## 可复用检索脚本

`retrieve_candidates.py`检索并保存候选；`verify_selection.py`核验入选元数据及定向全文证据。二者仅生成研究记录，不修改论文源文件。重新运行可能因数据库更新得到不同结果；当前JSON为本次检索快照。

## 后续调整：23篇与PDF归档

- 删除`peng2018dynamics`：主要面向机械臂控制，动力学随机化原理已有Tobin和四足专项研究Tan、Xie支撑，新增价值相对有限。
- 删除`rudin2022parallel`：重点为并行训练效率，与本文分层决策和执行裕度校准的关系较间接。相应训练提速句一并移除。
- 两条BibTeX条目及其正文引用均已删除，其他引用、方法和实验数据保留。原47条文献库现为45条，其中23条实际进入PDF。
- 初选10篇均下载为arXiv公开全文，8篇保留文献位于`papers/`，2篇已移除文献位于`removed_from_manuscript/`。后者仅作调研归档，不计入论文参考文献。
- [下载清单](../../../references/1029/literature_expansion_20261030/manifest.json)记录来源URL、文件大小、页数、SHA-256及首页文字片段。下载版本为预印本/作者公开稿，不冒充出版商最终排版版。
- `download_papers.py`可复核下载；10份文件均通过PDF打开及文字可读检查。
- 重新编译成功：7页、23篇参考文献，无缺失引用；首页仓库链接保留。末页参考文献已目视检查。仍有约1.73pt垂直溢出与epstopdf shell-escape提示，未发现可见遮挡或越界。

# 四足机器人导航研发工作区

本仓库集中管理 CaMRA 的实现、参考项目与论文资料。整体目标是在 Isaac Lab 中复现 Go2 分层最小成本到达避障。已有冻结 v2 控制器及其校准工具；当前新增 HiRA 预训练底层策略的接入验证与独立校准，之后再安排上层导航。

## 当前进展与下一步

当前任务是 [HiRA 策略接入与独立校准](doc/plan/20261105_camra_hira_policy_calibration.json)，共40项，当前 **38 项完成**，剩 HIRA-039（文档同步 + 只读稿件差异清单）与 HIRA-040（逐项对账与交接）。接入门禁、1820 条采集、α/δ 冻结、400 条独立检验、200 条近障评估、独立审计、与 v2 的报告级对比、复现包均已完成并留有报告。

本轮固定30秒测量窗口，成本采用正关节机械功（J）。计划独立完成20条pilot、400条功率基线、800条校准、400条独立检验和200条近障检查。HiRA 使用自己的功率模型、足迹、α/δ、数据锁和输出目录，不直接替换 v2。配额已按计划采完且业务失败零删除。HiRA 冻结值：α = 0.351568、δ = 9.032231 m；独立检验条件分母 377 条上成本 / 位置 / 联合覆盖为 0.9098 / 0.8966 / 0.8170。近障 δ 条件 **0/200**——δ 与 8×10 m 场地差一个量级，属如实上报的负结果。实际状态以计划 `tasks` 为准。

原 [v2校准计划](doc/plan/20261103_camra_calibration_implementation.json) 的32项任务已记录完成，工具、结果及限制继续保留。v2 的低速局限与旧底层复测未通过结论不变，具体证据见 [底层说明](CaMRA/locomotion/README.md#当前状态与已知边界) 和 [校准交接笔记](CaMRA/calibration/docs/implementation_handoff.md)。新增 HiRA 实验不默认重训底层或改写旧结论。

| 模块 | 当前状态 | 后续工作 |
| --- | --- | --- |
| [locomotion](CaMRA/locomotion/README.md) | v2 已冻结；HiRA 已接入并冻结（bundle `3c32fe31`） | 保留双套契约，不改旧模型与旧结论 |
| [calibration](CaMRA/calibration/README.md) | v2 与 HiRA 双套校准均已建立；HiRA 完成检验、近障、审计与交付包 | 供上层导航消费；稿件差异清单待作者执行 |
| [navigation](CaMRA/navigation/README.md) | 仅有用途说明，代码待实现 | 本轮不启动；后续消费匹配控制器的校准包 |

本阶段仅做仿真，执行主机的终端任务使用 tmux。HiRA 原系统使用 Isaac Gym，本轮目标是 Isaac Lab，不能仅换权重就认定可用；运行前须配置匹配环境并核对模型与资产。Agent 接手先读 [AGENTS.md](AGENTS.md)。

## 目录与阅读入口

```text
paper/
├── AGENTS.md          # Agent 接手入口：当前任务、阅读顺序、约束
├── README.md          # 工作区总览
├── CaMRA/             # 本项目实现
│   ├── locomotion/    # Go2 底层速度跟踪器
│   ├── calibration/   # 执行误差校准与独立检验
│   └── navigation/    # 上层最小成本到达避障导航器
├── RC-PPO/            # 冻结的原算法参考副本
├── HiRA/              # 固定只读子模块，本轮底层策略来源
├── manuscripts/       # 当前论文、历史稿件、文献、绘图及数据
├── doc/               # 实施计划与工作记录索引
└── .local/            # Git 忽略的本机备份
```

- [CaMRA 系统说明](CaMRA/README.md)：方法、模块关系和统一接口。
- [AGENTS.md](AGENTS.md)：接手任务时的最短阅读路径、配置依据与工作边界。
- [当前 HiRA 计划](doc/plan/20261105_camra_hira_policy_calibration.json)：40项任务、接入门禁、依赖与验收记录。
- [原 v2 校准计划](doc/plan/20261103_camra_calibration_implementation.json)：历史32项任务与执行证据。
- [文档索引](doc/README.md)：当前任务阶段及历史计划、执行证据入口。
- [RC-PPO](RC-PPO/README.md)、[HiRA](HiRA/README.md)：两个参考项目，各自的代码与使用说明。
- [论文资料](manuscripts/README.md)：稿件、图片、数据、参考文献及编译入口。

各目录的 README 解释该目录项目；详细实验经过和命令证据保留在对应计划、交接笔记中。

## 获取完整工作区

```bash
git clone --recurse-submodules https://github.com/haoyangc2001/Paper.git
```

已有检出目录在保存自身修改并更新主仓库后，初始化 HiRA：

```bash
git submodule sync -- HiRA
git submodule update --init --recursive -- HiRA
git submodule status -- HiRA
```

HiRA 使用主仓库锁定的提交，本阶段只读。若已有修改，先检查 `git -C HiRA status --short`；普通 clone 或主仓库 ZIP 不会带入子模块内容。HTTPS 受限且已有 SSH 权限时，可只覆盖本次连接地址：

```bash
git -c submodule.HiRA.url=git@github.com:haoyangc2001/HiRA.git submodule update --init --recursive -- HiRA
```

v2 权重、冻结包、外部资产和原始日志没有随主仓库克隆，取用与校验见 [底层复现说明](CaMRA/locomotion/docs/reproduction.md)。HiRA 子模块自带部分 checkpoint，但仍需按本轮计划核验指定权重、依赖和资产；已有文件不表示已完成 Isaac Lab 接入。

## 论文与保留资料

当前稿件：[1029 LaTeX](manuscripts/CCDC/latex/1029/main.tex)、[论文 PDF](manuscripts/CCDC/output/1029/main.pdf)、[历史写作说明](manuscripts/CCDC/README.md)。本轮资料只读；不改稿件、图表或 PDF，不重新编译。需要同步时仅输出差异清单交用户处理。

`manuscripts/CCDC/data/1029/` 保留数据及来源台账，`manuscripts/figures/` 保存各图的数据、脚本与成图。论文中的示例数值不作为本项目已完成实测的证据。

2026-10-31 的目录迁移映射见 [写作资料说明](manuscripts/README.md)；过程文件的本机可恢复备份见 [.local 清理说明](.local/cleanup_20261031/README.md)，该备份由 Git 忽略。历史稿件、原审稿意见和已有实验记录继续保留。

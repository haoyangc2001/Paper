# 四足机器人导航研发工作区

本仓库集中管理 CaMRA 的实现、参考项目与论文资料。整体目标是在 Isaac Lab 中复现 Go2 分层最小成本到达避障。后续开发固定使用冻结 v2 控制器；HiRA 接入与独立校准已完成，保留为历史对照资料。

## 当前进展与下一步

当前实施入口为 [固定 v2 的上层导航计划](doc/plan/20261109_camra_navigation_implementation.json)，版本 **1.2.0**，共 **60项**、当前 **0/60完成**。本轮仅修订计划，尚未实现导航。新增核查按依赖插入：先确定命令域、目标尺度、姿态几何和预算比较规则，再做低速与闭环裕度适用性检查，之后才进入上层学习；执行以计划数组顺序为准。

计划允许在既定范围回修前置实现：原记录和证据保留，修正过程追加到发现问题的当前任务，受影响的验收重验后再继续。完整规则见计划 `upstream_correction_policy`。

历史 [HiRA 策略接入与独立校准计划](doc/plan/20261105_camra_hira_policy_calibration.json)已记录 **40/40 项完成**。接入门禁、1820 条配额采集、α/δ 冻结、400 条独立检验、200 条近障评估、独立审计、与 v2 的报告级对比、复现包及最终交接均有报告；任务完成不表示全部科学指标达标。

该历史校准使用固定30秒测量窗口，成本采用正关节机械功（J）。20条pilot、400条功率基线、800条校准、400条独立检验和200条近障配额已采完，业务失败保留。HiRA 冻结值为 α = 0.351568、δ = 9.032231 m；独立检验在完整且可算的 377 条上覆盖率为成本 90.98% / 位置 89.66% / 联合 81.70%，按全体 400 条计分别为 85.75% / 84.50% / 77.00%。近障 δ 条件 **0/200**，该裕度在 8×10 m 场地中不可用；这不是导航成功率实验。

**后续开发硬约束（用户于2026-11-09确认）：固定使用 `camra_go2_velocity_it2999_v2`，并整套绑定其执行契约、资产与校准配置。** HiRA 仅作历史参考/对照；未经用户明确授权，不得换用 HiRA、其他 checkpoint 或重新训练替换 v2。完整约束见 [AGENTS.md](AGENTS.md#硬约束后续开发固定使用-v22026-11-09-用户确认)。

[控制器选择记录与复核说明](CaMRA/calibration/docs/controller_comparison_status_20261109.md)记录了 HiRA 批间初态、测量起点对齐、全体分母和交付索引待复核事项；它们不默认成为 v2 开发前置任务，共享代码问题按实际影响核查。迁移报告已确认 HiRA 数据复制到 tiger11；后续 v2 运行须另行核验自己的模型、资产、依赖和校准资源，不能由 HiRA 迁移结果替代。

原 [v2校准计划](doc/plan/20261103_camra_calibration_implementation.json) 的32项任务已记录完成，工具、结果及限制继续保留。v2 的低速局限与旧底层复测未通过结论不变，具体证据见 [底层说明](CaMRA/locomotion/README.md#当前状态与已知边界) 和 [校准交接笔记](CaMRA/calibration/docs/implementation_handoff.md)。新增 HiRA 实验不默认重训底层或改写旧结论。

| 模块 | 当前状态 | 后续工作 |
| --- | --- | --- |
| [locomotion](CaMRA/locomotion/README.md) | v2 已冻结；HiRA 历史包保留 | 后续固定使用 v2，不替换模型与执行契约 |
| [calibration](CaMRA/calibration/README.md) | v2 与 HiRA 双套历史校准均已建立 | 主线使用 v2 配套校准，HiRA 复核独立保留；稿件由作者修改 |
| [navigation](CaMRA/navigation/README.md) | 代码待实现；60项计划已创建并校验，0/60完成 | 从 NAV-001 学习背景，按依赖实现并验证 |

本阶段仅做 Isaac Lab 仿真，执行主机固定 tiger11（192.168.20.11），终端任务使用 tmux。先验证确定性名义环境和 v2 接口，再训练上层，最后固定策略做 clean 与扰动检验。运行前按计划核验环境、模型和资产；Agent 接手先读 [AGENTS.md](AGENTS.md)。

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
├── HiRA/              # 固定只读子模块，历史控制器对照及分层结构参考
├── manuscripts/       # 当前论文、历史稿件、文献、绘图及数据
├── doc/               # 实施计划与工作记录索引
└── .local/            # Git 忽略的本机备份
```

- [CaMRA 系统说明](CaMRA/README.md)：方法、模块关系和统一接口。
- [AGENTS.md](AGENTS.md)：接手任务时的最短阅读路径、配置依据与工作边界。
- [当前导航计划](doc/plan/20261109_camra_navigation_implementation.json)：60项任务、前置对齐核查、依赖与验收标准；尚未执行。
- [历史 HiRA 计划](doc/plan/20261105_camra_hira_policy_calibration.json)：40项任务、接入门禁、依赖与验收记录。
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

v2 权重、冻结包、外部资产和原始日志没有随主仓库克隆，取用与校验见 [底层复现说明](CaMRA/locomotion/docs/reproduction.md)。HiRA 指定 checkpoint 随子模块提供，独立冻结包在实验目录中；新机器仍需核验字节、外部日志、依赖和资产，详见 [HiRA 复现说明](CaMRA/calibration/experiments/hira_go2_18500/docs/reproduction.md)及最新复核记录。

## 论文与保留资料

当前稿件：[1029 LaTeX](manuscripts/CCDC/latex/1029/main.tex)、[论文 PDF](manuscripts/CCDC/output/1029/main.pdf)、[历史写作说明](manuscripts/CCDC/README.md)。本轮资料只读；不改稿件、图表或 PDF，不重新编译。需要同步时仅输出差异清单交用户处理。

`manuscripts/CCDC/data/1029/` 保留数据及来源台账，`manuscripts/figures/` 保存各图的数据、脚本与成图。论文中的示例数值不作为本项目已完成实测的证据。

2026-10-31 的目录迁移映射见 [写作资料说明](manuscripts/README.md)；过程文件的本机可恢复备份见 [.local 清理说明](.local/cleanup_20261031/README.md)，该备份由 Git 忽略。历史稿件、原审稿意见和已有实验记录继续保留。

# 项目文档与实施记录

本目录保存实施计划及其逐任务工作记录。工作区总览见 [根 README](../README.md)，Agent 当前任务与阅读路径见 [AGENTS.md](../AGENTS.md)；模块使用方法放在各自 README。

## 现有计划

[20261103_camra_calibration_implementation.json](plan/20261103_camra_calibration_implementation.json) 是当前校准模块的实施计划，版本1.1.0，共32项任务，尚未开始执行。按新机器、新模型上下文安排：前四项详细学习项目与论文背景并核对理解，随后三项实际配置环境、获取冻结模型和验证最小闭环，再进行能量协议、模块实现、独立功率建模、800条校准、400条独立检验和200条近障检查，最后复算并交付。

[20261031_camra_locomotion_implementation.json](plan/20261031_camra_locomotion_implementation.json) 记录底层实现的 28 项原任务和 9 项补充任务。阅读时先看 `implementation_clarifications`、`external_learning_report`，再按需查看任务的 `work_records`。

当前历史状态为 TASK-001–035 已完成，TASK-036/037 保持 `false`：确认性复测虽已执行，但未满足逐变体验收，收尾验收也未通过。具体证据见 [底层交接笔记](../CaMRA/locomotion/docs/implementation_handoff.md) §37–38。

## 当前安排

2026-11-03 用户已接受冻结 v2 的已知低速局限，下一步转向 [仿真执行误差校准与独立检验](../CaMRA/calibration/README.md)，不再默认继续底层调参。此决定不修改旧协议、旧失败结论或任务状态；历史文档中的等待修复/裁决安排按其生成时的上下文理解。

校准模块尚未实现，后续按新计划执行。已确定固定30秒测量窗口，增加名义功率模型与执行能量测量；纯时间不作本版成本。具体机械能量指标、基线采样预算和准入阈值在 CAL-008 核验后确认。功率基线数据独立于800/400/200条正式数据，旧论文的代理成本与预算不能直接当成能量结果使用。

执行或更新 JSON 计划时使用 `create-plan-worklog` 技能，保留依赖、验收证据和追加式工作记录。执行主机上的终端任务统一使用 tmux，首次建立 tmux 所需的最小引导命令、本地计划维护与结构校验除外。

## 其他资料的位置

- [CaMRA 模块说明](../CaMRA/README.md)：系统设计及模块接口。
- [底层交接笔记](../CaMRA/locomotion/docs/implementation_handoff.md)：历史命令、测试和诊断。
- [底层产物索引](../CaMRA/locomotion/docs/artifacts_index.json)：仓库外模型、日志与资产。
- [论文写作资料](../manuscripts/README.md)：稿件、图表、参考文献及数据来源。

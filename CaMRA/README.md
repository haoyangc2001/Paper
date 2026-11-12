# CaMRA

面向 Unitree Go2 的分层最小成本到达避障系统。冻结的底层策略负责速度跟踪、平衡与步态协调；执行校准模块测量剩余模型—执行偏差；上层导航利用成本与几何裕度生成到达目标的速度指令。

## 当前范围与状态

当前在 Isaac Lab 中逐步复现系统。`camra_go2_velocity_it2999_v2` 已冻结，校准工具及其历史结果已建立；HiRA 预训练底层策略也已完成接入、独立校准和交接。底层不默认继续训练或更换；当前工作是上层原理对齐、有限名义训练与闭环重验，真机通信、sim2real 与实机实验留待后续。

当前实施入口为[67项上层导航计划](../doc/plan/20261109_camra_navigation_implementation.json)，版本 **1.3.1**，**48/67项记录完成**，下一项 **NAV-061**。名义环境、三网络训练、预算工具和物理接入已存在，但旧三轮开发集成功率为0、G4为 `FAIL_WITH_STABILITY`；历史结案不等于性能通过。

本轮只原位修订计划及文档：NAV-061–063先恢复来源、修正时序/预算/终止与共同停止目标，NAV-064–066有限训练、重验部署actor/双critic并做独立价值预算搜索，NAV-067修正clean参数和200 Hz事件后重验接线，再进入正式评估。任务数、旧记录和状态保留，没有启动新实验。

历史 [40项 HiRA 计划](../doc/plan/20261105_camra_hira_policy_calibration.json)已记录 **40/40 项完成**。`locomotion/adapters/hira_go2/` 与 `calibration/experiments/hira_go2_18500/` 均已建立：前者 13 个模块，后者含配置、七清单、冻结包、报告与交付索引；运行入口与复现步骤见该实验 `docs/reproduction.md`。

**后续开发固定使用 `camra_go2_velocity_it2999_v2`（用户2026-11-09确认，硬性约束）。** 底层、校准和导航均绑定 v2 的模型、资产、执行契约及配套功率模型、足迹、α/δ和数据锁；未经用户明确授权，不切换到 HiRA 或其他 checkpoint，不重新训练替换 v2。HiRA 仅保留为历史参考/对照，完整约束见[根 AGENTS](../AGENTS.md)。

[控制器选择与复核记录](calibration/docs/controller_comparison_status_20261109.md)保留历史 HiRA 问题；它们不默认构成 v2 开发前置任务，涉及共享代码时核查对 v2 的影响。新版部署策略在正式物理评估前重新核验v2资源、接口与校准适用性，本次不启动训练。

| 模块 | 责任 | 输入 → 产物 | 状态 |
| --- | --- | --- | --- |
| [locomotion](locomotion/README.md) | 速度跟踪与稳定运动 | 机身速度指令、本体状态 → 12 维关节位置偏移 | v2 已冻结；HiRA 已接入并冻结（`3c32fe31`） |
| [calibration](calibration/README.md) | 能量与位置误差校准、独立检验 | 固定控制器、独立基线及配对轨迹 → 功率模型、α、δ与检验报告 | v2 与 HiRA 双套已建立；HiRA 完成检验/近障/审计/复现包 |
| [navigation](navigation/README.md) | 安全、到达与成本决策 | 地图/目标/状态、预算及裕度 → 三维速度指令 | 已有实现；48/67项记录完成，从NAV-061回修与重验 |

底层补充复测的历史结论仍为 `overall_pass=False`，低速扫描也发现起步死区。当前决定是在保留这些边界的前提下推进研究；详细数值和证据统一见 [底层说明](locomotion/README.md#当前状态与已知边界)。

## 系统流程

1. **固定执行系统。** 后续统一使用冻结 v2 的模型、观测与动作映射、增益、频率和资产。HiRA 独立冻结包及历史结果保留，不接替 v2。
2. **消费已冻结能量基线。** 原校准已用独立clean数据建立功率模型，并在同指令、同初态和固定30秒窗口下比较名义与物理执行轨迹。当前导航消费 v2 配套模型和数据锁，不默认重采或重新拟合。
3. **核验校准适用性。** 历史校准使用20条pilot、400条功率基线、800条校准、400条独立检验与200条近障配额。当前先检查速度域、低速执行及闭环/提前终止是否适用，保持已有 α、δ 冻结；不同控制器的结果不能混用。
4. **实现、训练和评估导航。** 先统一成本、预算、安全与终止规则并实现确定性名义环境，提前接通 v2 验证时序和计量，再训练策略与两个评价网络。完整方法按 `(1+α)c` 扣预算、在足迹之外额外计入 δ；部署策略及预算规则冻结后，对照无校准方法完成 clean 与扰动检验。首轮不默认增加消融或随机再训练。

名义轨迹来自导航模型；关闭随机扰动后的 Go2 仿真仍属于执行系统，仍可能存在跟踪误差。经验裕度需要与控制器版本、采样条件及检验范围一起解释，更换控制器或执行配置后重新校准。

## 统一运行接口

上层输出机身坐标系速度 `[vx, vy, yaw_rate]`；底层自行构造本体观测并输出关节位置偏移。校准模块离线生成裕度，上层与底层在线闭环执行。

当前冻结契约为仿真 200 Hz、底层策略 50 Hz、外部命令 10 Hz。底层每 0.02 s 重新观测并推理，上层命令每 0.1 s 更新；不能将一次关节动作保持 0.1 s 代替底层闭环。

v2 执行参数以其冻结包 `params/deploy.yaml`、`params/export_manifest.json` 和 [清单副本](locomotion/docs/frozen/camra_go2_velocity_it2999_v2.frozen_manifest.json) 为依据。HiRA 的缩放、默认姿态、关节顺序和显式PD须独立核验，不能沿用v2字段。论文历史5 Hz设定与当前实现的差异只记录，稿件保持只读；本轮差异清单见 `calibration/experiments/hira_go2_18500/reports/manuscript_sync_differences.md`。执行主机终端任务使用tmux。

两个控制器的契约不可互换（完整字段见各自 `controller.json`）：

| 项 | v2 `camra_go2_velocity_it2999_v2` | HiRA `hira_go2_18500` |
| --- | --- | --- |
| 来源 | unitree_rl_lab 训练后导出 | Isaac Gym 预训练权重 `model_18500.pt` 导出 |
| 观测缩放 | `base_ang_vel×0.2`、`joint_vel×0.05` | `ang_vel×0.25`、`command×[2,2,0.25]`、`dof_vel×0.05`，噪声只落 9:33 |
| 关节顺序 | 按关节类型分组（hip→thigh→calf） | 源 URDF 逐腿；目标按类型，需置换 `[0,3,6,9,1,4,7,10,2,5,8,11]` |
| 增益 | kp 25.0、kd 0.5 | kp 20.0、kd 0.5、`action_scale` 0.25 |
| 力矩限幅 | 23.7 | 逐关节 23.7，**calf 35.55** |
| 冻结裕度 | α 0.19048、δ 2.6139 m | α 0.351568、δ 9.032231 m |

## 使用与资料入口

- [底层 README](locomotion/README.md)：冻结包、已知边界、环境与运行命令。
- [校准 README](calibration/README.md)：现有数据链路、新候选隔离及裕度检验。
- [当前导航计划](../doc/plan/20261109_camra_navigation_implementation.json)：67项任务、原理对齐、门禁和回修规则；当前48/67项记录完成。
- [历史 HiRA 计划](../doc/plan/20261105_camra_hira_policy_calibration.json)：已完成的接入、校准与逐项记录。
- [原 v2 校准计划](../doc/plan/20261103_camra_calibration_implementation.json)：历史32项任务与证据，不作为本轮起点。
- [导航 README](navigation/README.md)：上层结构及与校准、底层的连接。
- [底层复现说明](locomotion/docs/reproduction.md)：模型、日志及资产的获取与校验。
- [官方现成控制方案核查](locomotion/docs/controller_options.md)：保留 2026-11-03 的调研记录。
- [实施交接笔记](locomotion/docs/implementation_handoff.md)：历史命令、测试与诊断证据；其中等待修复或裁决的表述反映当时状态，当前工作方向以本页及根 AGENTS 为准。

## 来源与参考项目

底层基于宇树官方 [unitree_rl_lab](https://github.com/unitreerobotics/unitree_rl_lab)，来源与许可见 [UPSTREAM.md](locomotion/UPSTREAM.md) 及模块内原始许可证。Python 包名保持兼容，Go2 任务 ID 为 `CaMRA-Go2-Velocity`。

[RC-PPO](../RC-PPO/README.md) 是冻结的最小成本到达避障算法参考；[HiRA](../HiRA/README.md) 是固定版本的分层 Reach-Avoid 参考子模块，均在本阶段只读。论文与数据来源见 [manuscripts](../manuscripts/README.md)。

CaMRA 的三个模块由工作区根 Git 仓库统一管理。

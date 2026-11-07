# CaMRA

面向 Unitree Go2 的分层最小成本到达避障系统。冻结的底层策略负责速度跟踪、平衡与步态协调；执行校准模块测量剩余模型—执行偏差；上层导航利用成本与几何裕度生成到达目标的速度指令。

## 当前范围与状态

当前在 Isaac Lab 中逐步复现系统。`camra_go2_velocity_it2999_v2` 已冻结，校准工具及其历史结果已建立；当前新增 HiRA 预训练底层策略的接入验证与独立校准。继续训练或调参不作为默认任务，真机通信、sim2real 与实机实验留待后续。

当前执行入口为 [40项 HiRA 计划](../doc/plan/20261105_camra_hira_policy_calibration.json)，**38/40 项完成**，剩 HIRA-039（文档同步与只读稿件差异清单）与 HIRA-040（逐项对账与交接）。`locomotion/adapters/hira_go2/` 与 `calibration/experiments/hira_go2_18500/` 均已建立：前者 13 个模块，后者含配置、七清单、冻结包、报告与交付索引；运行入口与复现步骤见该实验 `docs/reproduction.md`。

| 模块 | 责任 | 输入 → 产物 | 状态 |
| --- | --- | --- | --- |
| [locomotion](locomotion/README.md) | 速度跟踪与稳定运动 | 机身速度指令、本体状态 → 12 维关节位置偏移 | v2 已冻结；HiRA 已接入并冻结（`3c32fe31`） |
| [calibration](calibration/README.md) | 能量与位置误差校准、独立检验 | 固定控制器、独立基线及配对轨迹 → 功率模型、α、δ与检验报告 | v2 与 HiRA 双套已建立；HiRA 完成检验/近障/审计/复现包 |
| [navigation](navigation/README.md) | 安全、到达与成本决策 | 地图/目标/状态、预算及裕度 → 三维速度指令 | 代码待实现；本轮不启动 |

底层补充复测的历史结论仍为 `overall_pass=False`，低速扫描也发现起步死区。当前决定是在保留这些边界的前提下推进研究；详细数值和证据统一见 [底层说明](locomotion/README.md#当前状态与已知边界)。

## 系统流程

1. **固定执行系统。** 每个候选分别固定模型、观测与动作映射、增益、频率和资产。v2 保持冻结；HiRA 须先完成来源与接入验证，再形成独立冻结包。
2. **建立能量基线并采集配对轨迹。** 用独立clean数据建立并冻结名义功率模型。在同一指令序列和固定30秒窗口下，由确定性导航运动模型生成名义轨迹，由Go2仿真与冻结策略生成执行轨迹；同时计算预测能量、测量执行能量并记录实际扰动。
3. **校准并独立检验。** 用20条pilot验证采集链路，独立采集400条clean功率基线（240/80/80）。再用800条校准试验确定 α、δ，冻结后完成400条独立检验与200条近障检查。成本为正关节机械功，单位 J；不同控制器的模型、足迹和裕度不能混用。
4. **后续训练和评估导航。** 按 `(1+α)c` 扣减预算，将危险集合额外膨胀 δ，训练导航策略及两个评价网络，再完成对比与消融。该阶段另行安排。

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
- [当前 HiRA 计划](../doc/plan/20261105_camra_hira_policy_calibration.json)：本轮任务、依赖、接入门禁和逐项记录。
- [原 v2 校准计划](../doc/plan/20261103_camra_calibration_implementation.json)：历史32项任务与证据，不作为本轮起点。
- [导航 README](navigation/README.md)：上层结构及与校准、底层的连接。
- [底层复现说明](locomotion/docs/reproduction.md)：模型、日志及资产的获取与校验。
- [官方现成控制方案核查](locomotion/docs/controller_options.md)：保留 2026-11-03 的调研记录。
- [实施交接笔记](locomotion/docs/implementation_handoff.md)：历史命令、测试与诊断证据；其中等待修复或裁决的表述反映当时状态，当前工作方向以本页及根 AGENTS 为准。

## 来源与参考项目

底层基于宇树官方 [unitree_rl_lab](https://github.com/unitreerobotics/unitree_rl_lab)，来源与许可见 [UPSTREAM.md](locomotion/UPSTREAM.md) 及模块内原始许可证。Python 包名保持兼容，Go2 任务 ID 为 `CaMRA-Go2-Velocity`。

[RC-PPO](../RC-PPO/README.md) 是冻结的最小成本到达避障算法参考；[HiRA](../HiRA/README.md) 是固定版本的分层 Reach-Avoid 参考子模块，均在本阶段只读。论文与数据来源见 [manuscripts](../manuscripts/README.md)。

CaMRA 的三个模块由工作区根 Git 仓库统一管理。

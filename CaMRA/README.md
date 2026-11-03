# CaMRA

面向 Unitree Go2 的分层最小成本到达避障系统。冻结的底层策略负责速度跟踪、平衡与步态协调；执行校准模块测量剩余模型—执行偏差；上层导航利用成本与几何裕度生成到达目标的速度指令。

## 当前范围与状态

当前在 Isaac Lab 中完成仿真复现。已接受 `camra_go2_velocity_it2999_v2` 的已知低速局限，下一步实现执行误差校准与独立检验；底层训练代码保留供后续使用，继续调参不作为当前前置任务。真机通信、sim2real 适配与实机实验留待后续。

| 模块 | 责任 | 输入 → 产物 | 状态 |
| --- | --- | --- | --- |
| [locomotion](locomotion/README.md) | 速度跟踪与稳定运动 | 机身速度指令、本体状态 → 12 维关节位置偏移 | v2 已冻结，按已知边界复用 |
| [calibration](calibration/README.md) | 执行误差测量与独立检验 | 固定控制器、名义与执行轨迹 → α、δ 及检验报告 | 代码待实现；当前下一模块 |
| [navigation](navigation/README.md) | 安全、到达与成本决策 | 地图/目标/状态、预算及裕度 → 三维速度指令 | 代码待实现；在校准之后接入 |

底层补充复测的历史结论仍为 `overall_pass=False`，低速扫描也发现起步死区。当前决定是在保留这些边界的前提下推进研究；详细数值和证据统一见 [底层说明](locomotion/README.md#当前状态与已知边界)。

## 系统流程

1. **固定执行系统。** 使用同一 v2 模型、观测与动作映射、增益、频率及机器人资产。运行前取得仓库外冻结包并校验。
2. **采集配对轨迹。** 对同一指令序列，用确定性导航运动模型生成名义轨迹，用 Go2 仿真和冻结策略生成执行轨迹，记录实际施加的扰动。
3. **校准并独立检验。** 从相对成本超支与最大位置偏差的样本分位数确定 α、δ，再用分离的检验数据检查覆盖情况。
4. **训练和评估导航。** 按 `(1+α)c` 扣减预算，将危险集合额外膨胀 δ，训练导航策略及两个评价网络，再完成对比与消融。

名义轨迹来自导航模型；关闭随机扰动后的 Go2 仿真仍属于执行系统，仍可能存在跟踪误差。经验裕度需要与控制器版本、采样条件及检验范围一起解释，更换控制器或执行配置后重新校准。

## 统一运行接口

上层输出机身坐标系速度 `[vx, vy, yaw_rate]`；底层自行构造本体观测并输出关节位置偏移。校准模块离线生成裕度，上层与底层在线闭环执行。

当前冻结契约为仿真 200 Hz、底层策略 50 Hz、外部命令 10 Hz。底层每 0.02 s 重新观测并推理，上层命令每 0.1 s 更新；不能将一次关节动作保持 0.1 s 代替底层闭环。

执行参数以取得的冻结包 `params/deploy.yaml`、`params/export_manifest.json` 为依据，仓库内 [冻结清单副本](locomotion/docs/frozen/camra_go2_velocity_it2999_v2.frozen_manifest.json) 用于预先核对。论文中的历史 5 Hz 上层设定与当前实现不同，后续实验记录采用实际契约，并显式记录差异。

## 使用与资料入口

- [底层 README](locomotion/README.md)：冻结包、已知边界、环境与运行命令。
- [校准 README](calibration/README.md)：待实现的数据链路、裕度定义与交付要求。
- [导航 README](navigation/README.md)：上层结构及与校准、底层的连接。
- [底层复现说明](locomotion/docs/reproduction.md)：模型、日志及资产的获取与校验。
- [官方现成控制方案核查](locomotion/docs/controller_options.md)：保留 2026-11-03 的调研记录。
- [实施交接笔记](locomotion/docs/implementation_handoff.md)：历史命令、测试与诊断证据；其中等待修复或裁决的表述反映当时状态，当前工作方向以本页及根 AGENTS 为准。

## 来源与参考项目

底层基于宇树官方 [unitree_rl_lab](https://github.com/unitreerobotics/unitree_rl_lab)，来源与许可见 [UPSTREAM.md](locomotion/UPSTREAM.md) 及模块内原始许可证。Python 包名保持兼容，Go2 任务 ID 为 `CaMRA-Go2-Velocity`。

[RC-PPO](../RC-PPO/README.md) 是冻结的最小成本到达避障算法参考；[HiRA](../HiRA/README.md) 是固定版本的分层 Reach-Avoid 参考子模块，均在本阶段只读。论文与数据来源见 [manuscripts](../manuscripts/README.md)。

CaMRA 的三个模块由工作区根 Git 仓库统一管理。

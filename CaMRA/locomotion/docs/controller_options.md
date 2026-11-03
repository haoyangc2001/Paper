# 宇树官方现成控制方案核查

核查日期：2026-11-03。本记录从 CaMRA README 迁入，保留当时对官方公开仓库主线说明、Go2 配置及预训练目录的调研结果；后续仓库可能更新。

当时未找到同时配齐 Go2 仿真环境、Go2 预训练行走控制器和可直接调用的导航速度接口的完整开箱方案。

| 官方项目 | 已提供的部分 | 与当前需要的差距 |
| --- | --- | --- |
| [unitree_mujoco](https://github.com/unitreerobotics/unitree_mujoco/blob/main/readme.md) | Go2 物理仿真、机器人模型、底层控制及状态通信 | 官方说明当前仅支持低层开发；不包含真机内置 Sport 速度控制服务，不能仅靠 `SportClient.Move(vx, vy, wz)` 让仿真 Go2 行走 |
| [unitree_rl_lab](https://github.com/unitreerobotics/unitree_rl_lab) | Go2 速度跟踪训练环境及策略加载、控制框架 | [Go2 配置](https://github.com/unitreerobotics/unitree_rl_lab/blob/main/deploy/robots/go2/config/config.yaml) 的 `policy_dir` 指向训练日志；该配置目录未附现成策略，仍需取得匹配模型 |
| [unitree_rl_gym](https://github.com/unitreerobotics/unitree_rl_gym) | Go2 训练任务，以及部分机器人的 MuJoCo 推理示例和预训练模型 | 当时 [预训练目录](https://github.com/unitreerobotics/unitree_rl_gym/tree/main/deploy/pre_train) 与 [MuJoCo 配置目录](https://github.com/unitreerobotics/unitree_rl_gym/tree/main/deploy/deploy_mujoco/configs) 仅列 G1、H1、H1-2，不能作为 Go2 模型使用 |
| [unitree_sim_isaaclab](https://github.com/unitreerobotics/unitree_sim_isaaclab) | 带部分权重及移动操作入口的 Isaac Lab 仿真场景 | 当时说明列出的对象为 G1、H1-2，不是 Go2 的配套方案 |

因此沿用已有 Isaac Lab、冻结控制器与三维外部速度接口路线。当前使用决定与边界见 [locomotion README](../README.md#当前状态与已知边界)；本记录不替代包体取得、环境配置或执行设备上的运行验证。

现有闭环策略重载入口为 `scripts/rollout_exported_policy.py`，命令桥接为 `source/unitree_rl_lab/unitree_rl_lab/utils/external_command.py`，均相对于 locomotion 目录。官方调研本身不证明本机已运行这些入口。

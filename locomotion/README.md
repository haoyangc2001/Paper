# CaMRA Locomotion

CaMRA 的 Go2 底层强化学习速度跟踪模块。基于宇树官方 [unitree_rl_lab](https://github.com/unitreerobotics/unitree_rl_lab) 框架开展后续训练、评估、模型导出和部署适配。

底层控制器接收机身坐标系下的目标速度 `[vx, vy, yaw_rate]`，结合本体状态输出 12 个关节的位置调整量，由关节控制器执行。训练完成并通过评估后，控制器被冻结，供相邻的 `calibration/` 执行误差校准模块和 `navigation/` 上层导航模块使用。

## 来源与当前状态

- 本地项目名称为 **CaMRA Locomotion**，直接使用 `locomotion/` 作为项目根目录，不再套一层上游仓库目录。
- 已完整引入上游源码快照；本次仅新增本地说明，并将原始 README 保存为 [README.upstream.md](README.upstream.md)，训练和部署代码保持上游状态。
- 上游提交及引入范围见 [UPSTREAM.md](UPSTREAM.md)。保留原始 [LICENCE](LICENCE)、第三方许可和源码版权声明。
- Python 包名 `unitree_rl_lab`、启动脚本 `unitree_rl_lab.sh` 和任务 ID `Unitree-Go2-Velocity` 保留，以维持上游模块导入和脚本兼容性。这些内部标识不影响外层项目命名。
- 当前完成源码引入，尚未安装仿真依赖、下载机器人资产、训练 Go2 策略或接入上层速度指令。

## 主要入口

| 路径 | 用途 |
| --- | --- |
| `source/unitree_rl_lab/` | 可安装的 Python 扩展及环境实现 |
| `source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/go2/velocity_env_cfg.py` | Go2 速度跟踪环境、观测、动作和训练配置 |
| `source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/agents/rsl_rl_ppo_cfg.py` | PPO 训练配置 |
| `source/unitree_rl_lab/unitree_rl_lab/assets/robots/unitree.py` | 机器人资产路径及模型配置 |
| `scripts/rsl_rl/train.py`、`scripts/rsl_rl/play.py` | 训练和策略评估入口 |
| `deploy/robots/go2/` | Go2 的 C++ 部署入口和配置 |
| `doc/licenses/` | 上游附带的第三方许可证 |

上游其他机器人任务、部署依赖和示例模型一并保留，当前工作重点为 Go2 速度跟踪；其中其他机器人的示例模型不能用作 Go2 控制策略。

## 环境与启动

引入版本的上游 README 对应 **Isaac Lab 2.3.0 / Isaac Sim 5.1.0**。先按 [上游安装说明](README.upstream.md#installation) 准备兼容环境和机器人资产，并在 `unitree.py` 中配置实际资产路径。C++ 部署依赖和命令以 Linux 环境为基础，详见上游说明。

在已配置 Isaac Lab 的 Python 环境中，从本目录执行：

```bash
python -m pip install -e source/unitree_rl_lab
python scripts/list_envs.py
python scripts/rsl_rl/train.py --headless --task Unitree-Go2-Velocity
python scripts/rsl_rl/play.py --task Unitree-Go2-Velocity
```

评估命令需要已有训练检查点。这些是后续环境准备完成后的使用入口，本次未执行训练和仿真验证。

## 与上层的接口

上层只提供三维目标速度；底层负责组装关节状态、姿态、角速度和上一步动作等策略观测。后续接入时需要替换训练环境的随机指令生成器，统一坐标系、速度范围、关节顺序和动作缩放，并验证停车、低速转向及连续变速。模型、控制频率、控制增益和指令适配方式确定并冻结后，再开始执行误差校准。

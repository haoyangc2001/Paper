# CaMRA Locomotion

CaMRA 的 Go2 底层强化学习速度跟踪模块。基于宇树官方 [unitree_rl_lab](https://github.com/unitreerobotics/unitree_rl_lab) 框架完成训练、评估、导出与冻结。

底层控制器接收机身坐标系下的目标速度 `[vx, vy, yaw_rate]`，结合本体状态输出 12 个关节的位置调整量，由关节控制器执行。**本阶段已交付一个冻结控制器**，供相邻的 `calibration/` 执行误差校准模块和 `navigation/` 上层导航模块使用。

## 交付状态（2026-11-03）

| 阶段 | 结论 | 证据位置 |
| --- | --- | --- |
| 训练 | 第 4 轮调优，3000 迭代，选中 iteration 2999 | `…/2026-11-02_11-14-49_formal_s0_3000it_r4/model_2999.pt` |
| 开发集选模 | dev 种子 100–109，10 场景 × 300 回合全部通过 | 计划 TASK-023 工作记录 |
| 导出与数值验证 | JIT 1.19e-06、ONNX 1.12e-06（换算关节目标 2.79e-07 rad） | `$CAMRA_BUNDLE_ROOT/candidates/r4_it2999_v1/` |
| 独立闭环重载 | jit / onnx / trainer 三臂 4 场景 × 32 环境，0 摔倒 0 复位；JIT 与训练器逐位一致，ONNX 差 <9e-4 | `$CAMRA_LOG_ROOT/TASK-026/rollout/t026_aggregate.json` |
| **冻结交付件** | `camra_go2_velocity_it2999_v2`，29 文件、全树 0444/0555、校验 PASS | `$CAMRA_BUNDLE_ROOT/frozen/camra_go2_velocity_it2999_v2` |
| 最终测试（单次，旧口径） | final_test 种子 200–209：nominal 10/10、perturbed 10/10 场景通过；全局失败率 0.000 / 0.007。注意这是**场景级**口径，15 个协议变体里实际只测了 10 个（`lateral`/`turn_inplace` 各只下发一个方向、`combined` 只下发一个对角组合），见交接笔记 §37.2 | `$CAMRA_LOG_ROOT/TASK-024/` |
| **确认性复测（TASK-036）** | 新登记种子 300–309 × 每变体 300 回合 × 2 组 = **9000 回合**、指标有效数 100 %：**29/30 变体达标，`overall_pass=False`**。唯一不达标：nominal `lateral#v1`（`vy=−0.3`）`rmse_vy` 0.1035 > 0.1000（超 3.5 %，失败率 0，`p95` 0.1361 ≤ 0.18）；perturbed 同方向 0.1134 ≤ 0.1200 通过 | `$CAMRA_LOG_ROOT/TASK-036/report.md` |

**当前交付结论（TASK-036 补充复测后）**：冻结包 `camra_go2_velocity_it2999_v2` 仍是交付件，
但按补充协议 D-4 的**逐变体**判据**未通过验收**——15 个变体中 14 个在两组、300 回合、100 % 有效证据下达标，
唯一不达标的是 nominal 横向 `vy=−0.3 m/s` 的 `rmse_vy`（0.1035 > 0.1000，超 3.5 %，失败率 0，不掉不摔）。
因此**不得宣称已具备 calibration 交付条件**；处置选项见 [交接笔记 §37.6](docs/implementation_handoff.md)。
旧 TASK-024 的「nominal 10/10、perturbed 10/10」在旧口径下仍然成立，但它实测只覆盖了 15 个协议变体中的 10 个。

**接手请先读 [实施交接笔记 §30「阶段一交付与接手须知」](docs/implementation_handoff.md)**：交付物与校验入口、逐条真跑过的命令、后续校准必须保持一致的五件事、明确没做的事、以及踩过的坑。

仓库外的模型、原始日志和冻结包用 **[`docs/artifacts_index.json`](docs/artifacts_index.json)** 登记（路径 + sha256 / 目录树散列），
用 `python scripts/artifact_index.py verify` 一条命令核对；取用边界见 **[docs/reproduction.md](docs/reproduction.md)**。

## 来源与上游关系

- 本地模块名称为 **CaMRA Locomotion**，位于工作区的 `CaMRA/locomotion/`，不再套一层上游仓库目录；系统说明见 [CaMRA README](../README.md)。
- 已完整引入上游源码快照，并在其上做本项目改动；原始 README 保存为 [README.upstream.md](README.upstream.md)，上游提交及引入范围见 [UPSTREAM.md](UPSTREAM.md)。保留原始 [LICENCE](LICENCE)、第三方许可和源码版权声明。
- Python 包名 `unitree_rl_lab`、启动脚本 `unitree_rl_lab.sh` 保留，以维持上游模块导入和脚本兼容性；**任务 ID 已从上游的 `Unitree-Go2-Velocity` 改为本项目的 `CaMRA-Go2-Velocity`**，因为环境配置（奖励、课程、终止条件）已按本项目要求改写。上游示例模型不能用作 Go2 控制策略。
- 上游其他机器人任务、部署依赖一并保留，当前工作重点是 Go2 速度跟踪。

## 控制器契约（来自冻结包清单，不是文档转述）

| 项 | 值 |
| --- | --- |
| 观测 | 45 维，顺序 `base_ang_vel(3,×0.2)` → `projected_gravity(3)` → `velocity_commands(3)` → `joint_pos_rel(12)` → `joint_vel_rel(12,×0.05)` → `last_action(12)` |
| 动作 | `JointPositionAction`，12 维，`q_target = default_joint_pos + 0.25 · a`，wrapper 不裁剪动作 |
| 关节顺序 | FL/FR/RL/RR × hip/thigh/calf（仿真侧顺序；真机需 `joint_ids_map` 重映射，**未做**） |
| 控制频率 | sim 200 Hz（`dt=0.005`）× decimation 4 → 策略 50 Hz（`step_dt=0.02`）；外部命令 10 Hz 零阶保持 |
| 增益 | kp 25.0、kd 0.5、friction 0.01 |
| 命令域 | vx ±1.0 m/s、vy ±0.4 m/s、wz ±1.0 rad/s |
| 推理口径 | `torch.no_grad()`；**不用** `inference_mode`（会让仿真缓冲变成 inference tensor，复位时原地写入失败）；无经验归一化 |

## 主要入口

| 路径 | 用途 |
| --- | --- |
| `source/unitree_rl_lab/` | 可安装的 Python 扩展及环境实现 |
| `source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/go2/camra_env_cfg.py` | CaMRA 的 Go2 速度跟踪环境、观测、动作与奖励配置 |
| `source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/agents/rsl_rl_ppo_cfg.py` | PPO 训练配置 |
| `source/unitree_rl_lab/unitree_rl_lab/assets/robots/unitree.py` | 机器人资产路径及模型配置（读 `UNITREE_MODEL_DIR` / `CAMRA_ASSET_ROOT`） |
| `scripts/rsl_rl/train.py`、`scripts/rsl_rl/play.py` | 训练和策略评估入口（上游） |
| `scripts/eval_locomotion.py` | 按协议评估：逐场景指标、逐回合 CSV、summary JSON |
| `scripts/aggregate_eval_runs.py` | 跨种子/分组汇总与完整性检查（缺 run/场景/变体 → 退出码 2） |
| `scripts/recompute_metrics.py` | 从原始 npz 复算指标，与 per_episode.csv 对照 |
| `scripts/export_policy_candidate.py`、`scripts/verify_export_consistency.py` | 导出 JIT/ONNX 并做数值一致性验证 |
| `scripts/rollout_exported_policy.py`、`scripts/aggregate_rollout_arms.py` | 三臂闭环重载与等效性判定 |
| `scripts/freeze_controller_bundle.py`、`scripts/verify_frozen_bundle.py` | 生成只读冻结包并校验 |
| `scripts/external_command_drive.py` | 10 Hz 外部命令接口的独立驱动示例 |
| **`scripts/repro/`** | **仓库内复现入口**：`train` / `devsweep` / `finalize` / `finaltest` / `export` / `rollout`（见下） |
| `scripts/check_repro_readiness.py` | 预检：解释器、依赖、资产、检查点、协议、GPU，并区分「基础设施阻塞」与「控制器不合格」 |
| `scripts/artifact_index.py` | 仓库外产物的路径/散列索引构建与校验 |
| `config/repro.env` | 机器相关路径的唯一默认来源（环境变量优先） |
| `docs/eval_protocol.json` | 训练前固定的评估协议（种子、场景、阈值、验收规则） |
| `docs/eval_protocol_supplement_v1.0.0.md` | TASK-030 的四项裁决补充协议（单模型、每变体 300 回合、90% 有效下限、逐变体判定） |
| `docs/frozen/` | 冻结包清单副本（包体 28 MB 在仓库外） |
| `deploy/robots/go2/` | Go2 的 C++ 部署入口和配置（未做真机适配） |
| `tests/` | 纯 CPU 单元测试（指标、变体调度、有效性判定、跨种子聚合） |

## 环境与常用命令

实测环境：Python 3.11.18、torch 2.7.0（CUDA 12.6）、isaaclab 0.47.2、isaacsim 5.1.0.0、rsl-rl 3.0.1、onnxruntime 1.31.1。
上游 README 写的 Isaac Lab 2.3.0 / Isaac Sim 5.1.0 保留在 [README.upstream.md](README.upstream.md)，以实测版本为准。

**所有机器相关路径都在 [`config/repro.env`](config/repro.env) 里**（解释器、日志根、资产根、包根、设备）。换机器只改这一个文件，或直接导出同名环境变量（环境变量优先）：

```bash
export CAMRA_ASSET_ROOT=/mnt/assets/unitree_model
export CAMRA_LOG_ROOT=/mnt/logs/camra
python scripts/check_repro_readiness.py --level eval-now   # 先预检，别用失败的运行去评价控制器
```

```bash
cd CaMRA/locomotion

# 训练（默认 seed 0、3000 迭代；输出到 $CAMRA_LOG_ROOT/repro/train/）
scripts/repro/train.sh 0 3000 formal_s0_3000it_r4

# 开发集扫点 + 选模（输出到 $CAMRA_LOG_ROOT/repro/devsweep/）
scripts/repro/finalize.sh <run_dir> _r4

# 最终测试：2 分组 × 10 种子 × 每变体 300 回合（补充协议 D-2）
# 第 4 个参数是「每变体」而不是「每场景」：combined 有 4 个变体，写 30 会变成每变体 7 回合
scripts/repro/finaltest.sh <run_dir>/model_2999.pt both 200-209 300

# 导出 + 数值验证；三臂闭环重载 + 聚合判定
scripts/repro/export.sh <run_dir>/model_2999.pt
scripts/repro/rollout.sh <候选包> <run_dir>/model_2999.pt

# 冻结与校验（版本号不得复用；工作区不干净会被拒）
python scripts/freeze_controller_bundle.py --candidate <候选包> --run-dir <run_dir> \
  --checkpoint <run_dir>/model_2999.pt --version <新版本名>
python scripts/verify_frozen_bundle.py --bundle <冻结包> --samples 512

# 纯 CPU、随时可跑
python -m pytest tests/ -q
cd /tmp && python <仓库>/CaMRA/locomotion/scripts/verify_export_consistency.py --bundle <冻结包>
```

`scripts/repro/*.sh` 的每个脚本都会把命令行、主机、git 提交、脏文件清单、GPU 写进 `*.start.txt`；
产出判定以**产物是否存在**为准，不以 Isaac 进程退出码为准（该栈出现过 traceback 后仍 exit 0 的情况）。

完整命令（含参数含义、日志路径、恢复训练与校准入口）在 [交接笔记 §30.3](docs/implementation_handoff.md)，
换设备复现与产物取用边界在 [docs/reproduction.md](docs/reproduction.md)。

## 与上层的接口

上层只提供三维目标速度；底层负责组装关节状态、姿态、角速度和上一步动作等策略观测。冻结包已经把
观测顺序、动作缩放、关节顺序、控制频率、增益和命令域写进 `params/deploy.yaml` 与
`params/export_manifest.json`，两者是唯一权威来源——**不要从文档或记忆里取值**。

接入上层时仍需注意：真机 `joint_ids_map` 重映射、0–40 ms 指令延迟、状态估计噪声、执行器时延与
sim2real 标定**全部未做**；本包是仿真验证过的控制器，不是可直接上机的部署件。

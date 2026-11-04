# CaMRA 校准模块实施交接笔记

本文件是 `CaMRA/calibration/` 的持续更新学习与交接记录，按校准计划 CAL-001–004 的验收要求编写。
读者不应依赖任何历史聊天上下文，只需本文件、仓库文件与执行主机现场即可接手。

---

## 1. 研究目标与三模块数据流（CAL-002）

### 1.1 论文目标

当前主稿 `manuscripts/CCDC/latex/1029/main.tex`（CCDC，*Engineering Minimum-Cost Reach-Avoid Control
for Quadruped Navigation*）把四足导航表述为最小成本到达避障问题：

```text
min_{π,T}  Σ_{t=0}^{T-1} c(x_t, π(x_t))
s.t.       x_T ∈ G,  x_t ∉ F  ∀t ≤ T,  x_{t+1} ~ P(·|x_t, π(x_t))
```

按 RC-PPO 的做法把状态增广为 `(x, y, z)`：`y ∈ {-1,1}` 记录是否进过危险集，`z` 为剩余成本预算，
增广目标函数 `ĝ(x,y,z) = max{ g(x), C·y, -z }`，其中 `G={g≤0}`、`F={h>0}`、`C` 为违约常数。
组合残差 `W = max{ V_h, V_c - z/z_ref }`；Phase I 用 `W` 更新策略与两个评价网络，
Phase II 固定策略、重拟合评价网络并在候选初始预算中搜索满足可达/碰撞/预算条件的**最小**预算。

CaMRA 相对 RC-PPO 只增加两处**执行侧修正**：

```text
c̃(x_t,u_t) = (1+α)·c(x_t,u_t)            预算扣减按成本裕度放大
F_δ        = { x : d(x, F) ≤ δ }          危险集合按几何裕度膨胀
```

### 1.2 三个模块的数据流

```text
[导航 navigation]  状态/目标/危险集/剩余预算
        │  机身速度 u_t = [vx, vy, wz]          命令 10 Hz
        ▼
[底层 locomotion]  冻结 v2：45 维观测 → 12 维关节位置偏移
        │  仿真 200 Hz / 策略 50 Hz
        ▼
[物理执行]         Isaac Lab Go2 + 冻结策略（clean 或施加扰动）

[校准 calibration] 同一命令程序 + 同一 30 s 窗口 + 同一初始平面位姿
        ├── 名义侧：确定性平面运动模型 (px,py,yaw) 积分；能量由 P̂_nom 预测
        └── 执行侧：Go2 仿真真实执行；能量由 200 Hz 关节机械功积分
        ▼
   ε_c = (J_exec - J_nom)/J_nom     ε_d = max_t ‖p_exec(t) - p_nom(t)‖₂
        ▼
   α = Q_pc(ε_c),  δ = Q_pd(ε_d)   → 回灌 navigation（预算扣减与危险集膨胀）
```

各模块职责一句话：

| 模块 | 负责 | 不负责 |
| --- | --- | --- |
| locomotion | 平衡、步态、速度跟踪；提供稳定速度接口 | 决策、避障、成本 |
| calibration | 量化"名义—执行"差距，产出 α/δ 与功率模型 | 训练策略、改底层 |
| navigation | 到达、避障、成本最小化的决策 | 关节级控制、误差测量 |

---

## 2. 证据边界（必须最先理解）

### 2.1 论文数值全部是示例数据

`manuscripts/CCDC/data/1029/manuscript_completion_20261026/protocol.json` 自述：

```json
"data_status": "internal_synthetic_working_manuscript",
"empirical_validation": false
```

同目录 `README.md` 原文：**"全部新增配置及结果为 authored_example 或 computed_from_synthetic，不是实测"**，
并明确 **"不能把这些文件改名为 empirical 后当成采集证据"**。

因此下列论文数值**一律不得**作为本次仿真采样的结果、检验依据或达标证据：

- SR 73.3% / BVR 9.0%（未校准 RC-PPO 70.7% / 14.0%）
- α = 0.09、δ = 0.18、预算扫描约 0.27
- `z_ref` = 120 / 140 / 190 a.u.，900 trials/方法，800 校准 + 400 检验的样本计数
- 硬件对照 30 次（旧）与 50 次（新）

### 2.2 冻结控制器的历史状态

`camra_go2_velocity_it2999_v2` 的**旧验收未通过**：TASK-036 确认性复测 `overall_pass=False`，
唯一不达标项为 nominal `lateral#v1`（`vy=-0.3`）的 `rmse_vy` 0.1035 > 0.1000（超 3.5%）。
另有已测低速边界：约 ≤0.2 m/s 出现起步死区，0.3 m/s 附近稳态不足。
2026-11-03 用户决定**接受该局限继续校准**；接受不等于旧验收通过，旧记录保持原样。

### 2.3 可用与不可用的旧数据

| 数据类型 | 能否用于本次校准 |
| --- | --- |
| 论文/旧稿示例数值 | 否 |
| 底层 tracking RMSE（TASK-024/036） | 否，缺配对能量信号 |
| 冻结包 `samples/obs_samples.npz` | 否，仅用于导出数值一致性核对 |
| 新采的 clean 配对能量数据 | 是（唯一合法来源） |

---

## 3. 论文/旧稿与当前实现协议的差异（CAL-002）

### 3.1 已确认事实（有明确依据，不得再议）

| 项 | 论文 / 旧稿 | 当前实现协议 | 依据 |
| --- | --- | --- | --- |
| 成本语义 | 非负导航代理成本，**a.u.**，正文明确"not measured electrical energy" | **能量 J**：200 Hz 对关节正机械功积分 | 计划 `energy_contract`，用户 2026-11-03 决定 |
| 测量窗口 | episode 30 s，T 自由 | 固定 **30 s**（前 1 s 零命令 warmup 不计入） | `confirmed_protocol.measured_window_s` |
| 高层决策间隔 | **0.2 s（5 Hz）** | **0.1 s（10 Hz）** 外部命令 | 冻结契约；论文不改，记录差异 |
| 仿真/策略频率 | 50/200 Hz | 200 Hz 仿真、50 Hz 策略 | 冻结包 `params/deploy.yaml` |
| 样本划分 | 900 trials/方法（合成计数） | 800 校准 + 400 独立检验 + 200 近障，另有独立功率基线与 pilot | `confirmed_protocol.split` |
| 分位数 | 论文只写 `p_c`、`p_d` | 两项均 **90% nearest-rank** | `confirmed_protocol.quantiles` |
| 扰动条件 | 未细分 | 五 profile：clean / friction_only / mass_only / push_only / combined | `execution_profiles` |
| 命令域 | vx ±0.6、wz ±1.2 | vx ±1.0、vy ±0.4、wz ±1.0 | 用户 2026-10-31 决定 |
| α/δ 数值 | 0.09 / 0.18（示例） | **待实测**，禁止复用旧值 | `global_constraints` 第 10 条 |

### 3.2 待决事项（必须在 CAL-008/009 冻结，不得由实现者自行选择）

1. **具体机械能量指标**：推荐 `P = Σ_i max(τ_i·q̇_i, 0)`（关节间负功不互相抵消），
   备选 `Σ_i |τ_i·q̇_i|`、含负功损耗的电机能量、真实电功率。三者口径不同，选定后不得混用。
2. **`applied_torque` 的物理语义**：是否为 PD 限幅/摩擦后真正施加到关节的力矩，取步前还是步后时刻。
3. **功率模型形式与准入指标**：候选模型族、验证误差阈值、非负性与因果性约束。
4. **功率基线采样预算**：条数、拟合/开发/验证集划分、资源占用。
5. **零/近零 `J_nom` 规则**：`ε_c` 的分母可靠性阈值，禁止加任意小正数美化比值。
6. **指令延迟**：论文提及 0–40 ms 未建模、旧稿 100 ms；首版按计划排除，作为明确的未覆盖项记录。

---

## 4. 三个核心概念与对未来预算的影响（CAL-002）

### 4.1 确定性名义运动

状态只有平面 `(px, py, yaw)`；把**实际下发的同一命令**按航向旋转后积分。无物理引擎、无扰动、
无随机性，因此两次相同命令程序必须产生逐位一致的名义轨迹。
它代表"如果跟踪完美，机器人会走到哪里"，**不是**仿真结果。

### 4.2 clean 物理执行

关闭附加扰动的 Isaac Lab Go2 仿真 + 冻结 v2。即使 `friction=1`、`added_mass=0`、无推扰、无观测噪声，
仍存在接触求解、执行器限幅与策略跟踪误差，因此 clean **仍属于执行系统**，
不能称为名义模型。论文的 `J_exec` 对应这一类轨迹。

### 4.3 名义功率模型

用**独立 clean 基线数据**拟合的非负函数 `P̂_nom(命令[, 历史命令])`，令 `J_nom = Σ P̂_nom·Δt`。
它替代旧稿中"同一导航代理成本 `c` 同时作用于名义与实际两条轨迹"的写法。
按计划，`J_nom` 是**模型预测的机械能量**，`J_exec` 是**仿真关节机械功积分**，
两者使用同一能量指标与单位；但机械功不等于电池电能，报告时必须区分。

### 4.4 对新能量成本与上层预算的影响

1. 上层每 0.1 s 的步成本由 `c` 改为 `P̂_nom·Δt`，再按 `(1+α)` 扣减预算。
2. 若功率模型依赖历史命令，导航观测状态必须**增广**（否则模型输入不完整）。
3. 单位由 a.u. 变为 J，**旧稿所有预算数值与基准失效**，`z_ref` 与预算扫描需整体重算。
4. 定长 30 s 下纯时间成本会使 `α ≡ 0`（`J_nom = J_exec = 30`），这是本轮改用能量的直接原因。
5. 论文正文与图**不在本轮修改**；差异以本文件与后续报告记录。

---

## 5. 来源清单（本节随任务推进更新）

| 主题 | 文件 |
| --- | --- |
| 论文正文（方法、实验设定） | `manuscripts/CCDC/latex/1029/main.tex` |
| 数据性质声明 | `manuscripts/CCDC/data/1029/manuscript_completion_20261026/{protocol.json,README.md}` |
| 冻结契约 | 冻结包 `params/deploy.yaml`、`params/export_manifest.json`；仓库副本 `CaMRA/locomotion/docs/frozen/` |
| 执行协议 | `doc/plan/20261103_camra_calibration_implementation.json` 的 `confirmed_protocol`、`energy_contract` |
| 底层历史边界 | `CaMRA/locomotion/README.md`、`CaMRA/locomotion/docs/implementation_handoff.md` §10、§30、§37–47 |

---

## 6. 代码入口与调用链（CAL-003）

### 6.1 从速度指令到物理执行的调用链

| 步 | 动作 | 文件 / 符号 |
| --- | --- | --- |
| 1 | 生成命令程序 | `scripts/rollout_exported_policy.py::SCENARIOS` → `StepProgram` / `RampProgram` |
| 2 | 命令锁定与下发 | `unitree_rl_lab/utils/external_command.py`：`ExternalCommandWrapper.set_program` → `ExternalCommandBridge._install()` 包装 `CommandTerm.compute`，每个策略步 `_apply_locked()` 回写命令缓冲；`notify_step()` 产出 `CommandStepRecord`（applied / terminated / truncated / new_episode） |
| 3 | 组装观测 | `ManagerBasedRLEnv.observation_manager.compute()`，45 维顺序见 §6.3 |
| 4 | 策略推理 | `torch.jit.load(<bundle>/exported/policy.pt)` / `OnnxPolicy` / rsl-rl `OnPolicyRunner`（`build_policy()`），运行在 `torch.no_grad()` |
| 5 | 动作映射 | `JointPositionAction`：`processed = default_joint_pos + 0.25 · a`，wrapper 不裁剪动作（`clip_actions=None`） |
| 6 | 物理执行 | 每物理步（dt = 0.005）`UnitreeActuator.compute()`：`DelayedPDActuator.compute` 做 PD + 延迟缓冲 + `_clip_effort()`（力矩—转速曲线），再减去摩擦 `Fs·tanh(q̇/Va) + Fd·q̇`，最后 `control_action.joint_efforts = applied_effort` |
| 7 | 记录 | 仅在第 5 步（50 Hz）记录 `root_lin_vel_b`、`root_ang_vel_b`、`root_quat_w`、`root_pos_w`、`action`、`cmd` 存入 npz |

**关键事实（直接影响校准实现）**：第 7 步**没有任何力矩与关节速度记录**。
仓库内 `grep -rn "applied_torque"` 在 `scripts/`、`tests/` 下只命中
`export_policy_candidate.py` 中 `effort_limit_sim` 这个**配置字段名**，不是测量。
因此 200 Hz 的物理子步力矩必须由校准模块新采集，不能从旧产物反推。

### 6.2 力矩语义（`unitree_actuators.py`）

`UnitreeActuator.compute()` 的执行顺序是：

1. `self._joint_vel[:] = joint_vel`：保存**步前**关节速度，供力矩—转速曲线使用；
2. `super().compute()`：PD 计算 + 延迟缓冲 → 写入 `self.applied_effort`；
3. `self.applied_effort -= Fs·tanh(q̇/Va) + Fd·q̇`：在限幅之后的力矩上**扣除摩擦**；
4. `control_action.joint_efforts = self.applied_effort`：这才是送进物理引擎的力矩。

所以"施加力矩"应取 `ArticulationData.applied_torque`（= 上述 `applied_effort`），
**不是** PD 未限幅的 `computed_torque`，也不含力矩—转速曲线之外的任何额外项。
摩擦项可能使 `applied_effort` 与运动方向相反，这正是能量定义里负功处理的来源。

### 6.3 冻结契约（以冻结包 `params/deploy.yaml`、`export_manifest.json` 为准）

```text
观测 45 维 = base_ang_vel(3, ×0.2) → projected_gravity(3) → velocity_commands(3)
           → joint_pos_rel(12) → joint_vel_rel(12, ×0.05) → last_action(12)
动作 12 维 = default_joint_pos + 0.25 · a
关节顺序   = FL/FR/RL/RR hip → thigh → calf（按关节类型分组，完整顺序读清单 joint_names）
时序      = sim 200 Hz(dt 0.005) × decimation 4 → 策略 50 Hz(0.02) → 外部命令 10 Hz 零阶保持
增益      = kp 25.0 / kd 0.5 / friction 0.01；命令域 vx ±1.0、vy ±0.4、wz ±1.0
推理      = torch.no_grad()（**不用** inference_mode，否则仿真缓冲变 inference tensor、复位写入报错）
```

### 6.4 RC-PPO 参考副本的语义（`RC-PPO/`，只读）

| 项 | 源码行为 | 校准需注意 |
| --- | --- | --- |
| 入口 | `rl/EC-EFPPO.py`（内部命名 EC-EFPPO / EFPPO） | 目录名与代码命名不一致，属正常 |
| 网络 | `model/actorcritic.py`：独立 `Policy_Network` + 两个独立 critic（`train_state_h`、`train_state_energy`） | 不是合并式 actor-critic |
| 联合评价 | `max(V_h, V_energy - energy)` | `energy` 是**剩余成本预算**，不是电池能量 |
| 策略更新 | 对两个裁剪候选取 `maximum` 后**最小化** | 不能改写成普通 PPO 的奖励最大化 |
| 代价语义 | 环境返回非负逐步消耗并从预算扣除（接口仍叫 `reward`） | 名字叫 reward，语义是 cost |
| 终止 | `rl/gae.py` 重建的 `done` 用于学习目标分段 | 不等于环境返回的终止标志 |
| 预算搜索 | `rl/root_finding.py` 二分 | 当前训练入口未直接调用 |

与本次校准的关系：校准**只产出** α 与 δ 及功率模型，不改变 RC-PPO 的学习语义。
α 作用于运行成本与预算扣减，δ 作用于安全指示与可达目标，两者的作用点不同，不可互换。

### 6.5 HiRA 对照（`HiRA/`，只读子模块，锁定 6a05af5）

HiRA 是分层 Reach-Avoid 参考实现（Isaac Gym + legged_gym/rsl_rl 栈），提供"高层速度接口 +
冻结低层"的工程范式与训练/评估流程参考，其控制频率与观测配置见
`CaMRA/locomotion/docs/implementation_handoff.md` §13。
它**不实现** CaMRA 的预算条件双评价与执行数据校准；其训练脚本记录的 `cost`
是成功轨迹首次到达步数的均值，不是剩余预算状态，也不是单独的成本 critic 目标。
因此 HiRA 只作时序与流程参考，不能作为本次校准或上层成本定义的依据。

### 6.6 三项确认

1. **calibration / navigation 尚未实现**：两目录各只有 1 个 README、0 个 `.py`
   （见 `c1_device_baseline.txt` 的模块代码量统计）。
2. **模型与资产不随普通克隆齐备**：仓库内只有 `docs/frozen/*.frozen_manifest.json` 清单副本，
   28 MB 包体与 Go2 资产在仓库外，需按 `docs/reproduction.md` 取用并校验。
3. **旧跟踪日志缺正式配对能量信号**：仓库脚本无任何力矩采集（§6.1）；TASK-024 汇总
   （冻结包 `results/task024_aggregate.txt`）的指标只有 `fail / stand_drift_m / max_roll_deg /
   max_pitch_deg / rmse_* / p95_abs_err_* / rise_time_s / stop_drift_m / overshoot_mps`，
   **没有能量、力矩或成本字段**，也无名义—执行配对。

---

## 7. 文件与代码索引（CAL-004）

### 7.1 计划与说明

| 路径 | 作用 |
| --- | --- |
| `doc/plan/20261103_camra_calibration_implementation.json` | **当前执行计划**（32 项 CAL 任务，唯一状态来源） |
| `doc/plan/20261031_camra_locomotion_implementation.json` | 底层历史计划（TASK-036/037 保持 false，仅作证据） |
| `AGENTS.md` / `README.md` | Agent 接手入口 / 工作区总览 |
| `CaMRA/README.md` | 系统方法与三模块接口 |
| `CaMRA/{locomotion,calibration,navigation}/README.md` | 各模块现状、输入输出、边界 |
| `CaMRA/calibration/docs/implementation_handoff.md` | 本文件 |

### 7.2 底层代码（`CaMRA/locomotion/`）

| 路径 | 作用 |
| --- | --- |
| `source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/go2/camra_env_cfg.py` | CaMRA 环境 override 层 + `assert_overrides_applied` 运行时断言 |
| `.../go2/velocity_env_cfg.py` | 上游基线环境配置（被 override） |
| `.../locomotion/agents/camra_ppo_cfg.py` | CaMRA PPO 训练配置 |
| `.../mdp/commands/velocity_command.py` | 命令项（含 stop_event 等采样逻辑） |
| `.../mdp/{rewards,observations,curriculums,stop_events}.py` | 奖励、观测项、课程、停车事件 |
| `.../utils/external_command.py` | 外部速度命令桥接（`ExternalCommandBridge` / `CommandProgram` 族 / `CommandStepRecord`） |
| `.../utils/{eval_metrics,variant_scheduler,parser_cfg,export_deploy_cfg}.py` | 指标、变体调度、参数解析、导出配置 |
| `.../assets/robots/unitree.py` | 机器人资产路径（读 `UNITREE_MODEL_DIR` / `CAMRA_ASSET_ROOT`） |
| `.../assets/robots/unitree_actuators.py` | `UnitreeActuator`：力矩—转速曲线与摩擦模型 |

### 7.3 底层脚本（`CaMRA/locomotion/scripts/`）

| 路径 | 作用 |
| --- | --- |
| `rollout_exported_policy.py` | **闭环重载三臂**（jit/onnx/trainer）+ 接口自检；本次 CAL-007 用它 |
| `eval_locomotion.py` | 按协议逐场景评估，输出逐回合 CSV 与 summary JSON |
| `export_policy_candidate.py` / `verify_export_consistency.py` | 导出 JIT/ONNX 与数值一致性 |
| `freeze_controller_bundle.py` / `verify_frozen_bundle.py` | 冻结包生成与校验 |
| `check_repro_readiness.py` | 环境预检，区分 `infra_blocked(2)` 与 `env_incompatible(3)` |
| `artifact_index.py` | 仓库外产物的路径/散列索引与校验 |
| `recompute_metrics.py` / `aggregate_*.py` | 从原始 npz 复算与跨种子汇总 |
| `repro/*.sh` | 仓库内复现入口（train/devsweep/finalize/finaltest/export/rollout） |
| `tests/` | 纯 CPU 单元测试（指标、变体调度、有效性判定、聚合） |

### 7.4 协议与产物索引（`CaMRA/locomotion/docs/`）

| 路径 | 作用 |
| --- | --- |
| `eval_protocol.json` | 训练前固定的评估协议（场景、阈值、验收规则） |
| `eval_protocol_supplement_v1.0.0.md` | TASK-030 的四项裁决补充协议 |
| `frozen/camra_go2_velocity_it2999_v2.frozen_manifest.json` | 冻结包清单副本（包体在仓库外） |
| `artifacts_index.json` | 仓库外模型/日志/资产的路径与散列索引 |
| `reproduction.md` | 换设备取用与校验四步 |
| `implementation_handoff.md` | 底层 3581 行实施证据（§10 差异、§30 交付、§37–47 低速诊断） |

### 7.5 参考项目（只读）

| 路径 | 作用 |
| --- | --- |
| `RC-PPO/` | 冻结的最小成本到达避障参考实现（JAX；`rl/`、`model/`、`env/`） |
| `HiRA/` | 固定提交 `6a05af5` 的分层 Reach-Avoid 参考子模块（Isaac Gym 栈） |

---

## 8. 新设备配置清单与模型获取清单（CAL-004）

### 8.1 设备配置清单（本机已实测）

| 项 | 值 | 来源 |
| --- | --- | --- |
| 执行主机 | `tiger10.ios.ac.cn` / 192.168.20.10 | `hostname` |
| 工作区 | `/home/caohy/repositories/Paper`，HEAD `a92724f`，main，干净 | `git status` |
| 解释器 | `/pub/data/caohy/miniconda/envs/env_isaaclab/bin/python`（3.11.15） | 本机创建 |
| 关键包 | torch 2.7.0+cu126、isaacsim 5.1.0.0、isaaclab 0.47.2、assets 0.2.3、rl 0.4.4、tasks 0.11.6、rsl-rl-lib 3.0.1、numpy 1.26.0、onnx 1.23.2、onnxruntime 1.31.1 | `pip freeze` |
| Isaac Lab 源码 | `/pub/data/caohy/src/IsaacLab-2.3.0/{apps,source}`（五个扩展 editable） | 本机还原 |
| GPU | 3 × A100-PCIE-40GB，驱动 560.35.03（CUDA 12.6 上限） | `nvidia-smi` |
| 磁盘 | `/pub` 29T（剩 1.9T，94% 已用）；`/home/caohy` 20T（剩 16T） | `df -h` |
| tmux | 会话 `camra-env`（安装/仿真）、`camra-baseline`（连通性基线） | `tmux ls` |
| 日志根 | `/pub/data/caohy/logs/camra`、`/pub/data/caohy/logs/rsl_rl` | 目录存在 |
| 必要环境变量 | `OMNI_KIT_ACCEPT_EULA=YES`、`TMPDIR=/pub/data/caohy/tmp`、`XDG_DATA_HOME`/`XDG_CACHE_HOME` 指向 `/pub/data/caohy/xdg*` | 实测需要 |
| 安装与验证证据 | `/pub/data/caohy/logs/camra/env_setup/`（含 `README.md` 与 8 份日志） | 本机 |

**注意**：`config/repro.env` 中的 `CAMRA_*` 路径与本机布局一致，**未修改**；
其中 `CAMRA_TMUX_SESSION=camra-locomotion` 与实际会话名不同，预检只记为 warn，不阻塞。

### 8.2 模型与产物获取清单

历史服务器路径（`caohy@192.168.20.12`）**只作定位线索**，以下"本机落点"均已实测存在。

| 对象 | 历史路径（线索） | 本机落点 | 校验结果 |
| --- | --- | --- | --- |
| 冻结包 v2 | `/pub/data/caohy/camra_bundles/frozen/camra_go2_velocity_it2999_v2` | 同名路径 | 29 文件、树 `d0f0ec…`、清单 `6da14ce2…`，`verify_frozen_bundle.py` PASS |
| Go2 资产 | `/pub/data/caohy/unitree_model/Go2` | 同名（取自冻结包内拷贝） | 树 `9560507…`（`Go2/` 前缀规则）匹配清单 |
| 检查点 | `.../rsl_rl/camra_go2_locomotion/2026-11-02_11-14-49_formal_s0_3000it_r4/model_2999.pt` | 同名路径 | sha256 `1c8f5aad…` 与交接笔记 §30.1 一致 |
| 训练参数 | 同目录 `params/{agent,env}.yaml` | 同名路径 | `env.yaml` sha256 `922d473c…` 与 artifacts_index 一致 |
| 候选包 `r4_it2999_v1` | `.../camra_bundles/candidates/r4_it2999_v1` | **未取** | 预检 `candidate_bundle` 项报 ok（非阻塞），需要时再取 |
| TASK-024/026/036 原始日志 | `/pub/data/caohy/logs/camra/TASK-0xx` | **未取**（本机无） | 仅用于追溯历史判定，本次校准不依赖 |

取用通道说明：`.10 → .12` 无免密授权，实际采用经本地 Windows 中转的
`ssh .12 tar -cf - … | ssh .10 tar -xf -` 管道；`.12` 侧只读，未做任何远程写。

### 8.3 可直接执行的后续入口

```bash
# 预检（预期 VERDICT: ready）
<TMPDIR=/pub/data/caohy/tmp> /pub/data/caohy/miniconda/envs/env_isaaclab/bin/python \
  CaMRA/locomotion/scripts/check_repro_readiness.py --level eval-now

# 冻结包校验（预期 PASS）
<PY> CaMRA/locomotion/scripts/verify_frozen_bundle.py \
  --bundle /pub/data/caohy/camra_bundles/frozen/camra_go2_velocity_it2999_v2 --samples 512

# 最小闭环（预期 4 场景 fails=0 resets=0）
<PY> -u CaMRA/locomotion/scripts/rollout_exported_policy.py --headless \
  --task CaMRA-Go2-Velocity --bundle <冻结包> --backend jit \
  --num_envs 8 --seed 100 --out_dir <输出目录>
```

---

## 9. 能量信号审查（CAL-007）

本节记录在冻结 v2 上实测到的采集可行性，是 CAL-008 决定能量口径与 CAL-014 实现计量器的直接依据。

### 9.1 最小闭环

`rollout_exported_policy.py --backend jit`（8 envs、seed 100）在 4 个场景全部完成：

| 场景 | rmse_vx | rmse_vy | rmse_wz | fails | resets |
| --- | --- | --- | --- | --- | --- |
| stand | 0.0248 | 0.0166 | 0.0099 | 0 | 0 |
| start_stop | 0.0883 | 0.0358 | 0.0357 | 0 | 0 |
| turn | 0.0337 | 0.0254 | 0.1075 | 0 | 0 |
| ramp | 0.1201 | 0.0380 | 0.0523 | 0 | 0 |

接口自检 4/4 通过：`no_trainer_dependency`、`obs_dim`(45)、`obs_pipeline_reproducible`、
`action_scale_offset`（对 `default + 0.25·a` 误差 < 1e-5）。产物为 `trace_jit_<scenario>.npz`
（字段 `cmd/vx/vy/wz/roll/pos/actions`）与 `rollout_jit.json`。

**注意**：该脚本**不含**关节速度与力矩字段（见 §6.1），因此能量信号由下面的专用探针实测。

### 9.2 物理子步可采性（实测）

专用探针（`env_setup/09_energy_probe.py`，只读、不入库）在 2.0 s 窗口、2 envs 上：

| 项 | 结果 |
| --- | --- |
| 物理子步数 | 记录 400 = 2.0 s / 0.005 s，正好 200 Hz |
| 数据形状 | `applied_torque` (400, 2, 12)、`joint_vel` (400, 2, 12)、`root_pos` (400, 2, 2) |
| 关节顺序 | 与 `export_manifest.json` 的 `joint_names` **完全一致** |
| 数值有限性 | τ / q̇ / pos 全部 finite |
| 量级 | max abs τ = 15.78 N·m，max abs q̇ = 8.67 rad/s |
| 子步差异 | 第 1 个策略步内 4 个子步的 τ 最大差 5.42（并非重复值） |

### 9.3 关键陷阱：关节速度不是每子步刷新的

`ArticulationData.joint_vel` 是**时间戳门控的惰性读取**：

```python
if self._joint_vel.timestamp < self._sim_timestamp:
    self._joint_vel.data = self._root_physx_view.get_dof_velocities()
    self._joint_vel.timestamp = self._sim_timestamp
return self._joint_vel.data
```

`ArticulationData.update(dt)` 只做 `self._sim_timestamp += dt`。因此在 decimation 循环内
**不调用** `robot.update(sim_dt)` 时，`joint_vel` 整段保持上一次 `scene.update()` 的值。

实测对比（同一 2.0 s 窗口）：

| 取速方式 | 子步间是否变化 | 与 PhysX 直读是否一致 | 正功诊断值 (J) | 净功诊断值 (J) |
| --- | --- | --- | --- | --- |
| 不刷新（默认） | 否 | — | 0.0520 | −0.0228 |
| 每子步 `robot.update(sim_dt)` | 是（1.85 rad/s） | **逐位相同** | 0.0629 | −0.0024 |

即：用未刷新的 50 Hz 速度配合 200 Hz 力矩，会把正功**低估约 20%**，净功误差更大。
这与 `global_constraints` 第 14 条「50Hz 控制日志不直接冒充物理子步测量」完全对应。

**结论（给 CAL-014）**：能量计量若在 decimation 内积分，必须在每个 `sim.step` 之后
调用 `robot.update(sim_dt)` 再读 `joint_vel`；等价的直读是
`robot.root_physx_view.get_dof_velocities()`（实测两者逐位相同）。
`applied_torque` 本身每子步都会更新（见 §9.2 子步差异），不需要额外刷新。

### 9.4 步前 / 步后时刻

| 量 | 观测时点 | 说明 |
| --- | --- | --- |
| `applied_torque` | `sim.step()` 返回后 | 该物理步实际施加的力矩，等于 `UnitreeActuator` 写入的 `applied_effort`（§6.2） |
| `joint_vel`（post） | `sim.step()` + `robot.update(sim_dt)` 后 | 该步结束时的关节速度 |
| `joint_vel`（pre） | 该步 `sim.step()` 之前 | 上一物理步结束时的速度，即执行器 `_joint_vel` 用于力矩—转速曲线的值 |

两者在同一子步内可差到 13.44 rad/s，因此**口径必须显式冻结**（CAL-008），
不能在不同实现之间混用。

### 9.5 本次未能验证的项（如实记录，不当作通过）

1. 真实电机电能 / 电池功率：无电机与电池参数，仿真只给出机械功，**不能**标称为耗电。
2. 0–40 ms 指令延迟与观测噪声对能量的影响：首版协议排除，未建模。
3. 真机 `joint_ids_map` 重映射：未实现，本包仅仿真侧关节顺序。
4. ONNX 图固定 batch=1：未做批推理等价性验证。
5. 长窗口（30 s）连续采样的稳定性：本次只跑 2.0 s 探针，正式 30 s 见 CAL-018 小批验证。

---

## 10. D-ENERGY-001 裁定与能量成本规范（CAL-008）

用户于 2026-11-03 明确选择以下四项，本节为其可执行化定义；任何实现不得偏离，
需要变更必须回到 CAL-008 重新询问。

| 决策项 | 选定 |
| --- | --- |
| 主能量指标 | **正关节机械功** |
| 功率基线规模 | **400 条**（拟合 240 / 开发 80 / 验证 80） |
| 模型准入 | 验证集逐轨迹 `J_nom` 相对误差 **90% 分位 ≤ 10%** |
| 近零 `J_nom` | **绝对阈值 + 单列**，`J_min = 1.0 J`（30 s 窗口） |

### 10.1 主公式、符号与单位

```text
J_exec = Σ_{k=1..6000} Σ_{j=1..12} max( τ_j[k] · q̇_j[k] , 0 ) · 0.005      [ J ]
```

| 符号 | 含义 | 取值来源 |
| --- | --- | --- |
| `k` | 物理子步索引 | 30 s / 0.005 s = **6000** 步（不含 1 s warmup） |
| `τ_j[k]` | 第 k 步第 j 关节**实际施加力矩** | `robot.data.applied_torque`（= `UnitreeActuator.applied_effort`），在 `sim.step()` 返回后读取 |
| `q̇_j[k]` | 同一子步关节角速度 | `robot.data.joint_vel`，但**必须先 `robot.update(sim_dt)`** 才会刷新（§9.3）；等价写法 `root_physx_view.get_dof_velocities()` |
| `dt` | 物理步长 | 0.005 s（仿真 200 Hz） |

单位一律为 **焦耳（机械功）**。不得写作 Wh、kWh 或"电池耗电"。

### 10.2 三类边界情形的处理（无歧义）

| 情形 | 处理 |
| --- | --- |
| **站立 / 近零运动** | `q̇≈0` 使 `P≈0`，`J_exec` 接近 0；此类样本按 §10.4 的近零规则处理，**不**人为抬高 |
| **负功** | 由 `max(·,0)` 逐关节截断后求和，**不进入** `J_exec`，因此 `J_exec ≥ 0`；不要先求和再截断（会让关节间负功抵消） |
| **外部推扰** | 推扰对机器人做功不来自关节电机，**不计入** `J_exec`；只在 trial 元数据里记录 push 事件时刻与幅度，作为条件标签 |
| **摩擦项** | `applied_effort` 已扣除摩擦（§6.2），属于"施加到关节的力矩"，直接使用，不重复扣减 |

注意：`max(·,0)` 是**逐关节**再求和。写成 `max(Σ_j τ_j q̇_j, 0)` 会改变语义，禁止。

### 10.3 名义侧：功率模型与 `J_nom`

```text
P̂_nom = f(当前命令, 最近 K 条命令)  ≥ 0
J_nom = Σ_{k=1..6000} P̂_nom(·) · 0.005      [ J ]
```

- `f` 只在**独立 clean 基线数据**上拟合（见 §10.5），不接触 800/400/200 正式数据；
- 若 `f` 输出负值，截断为 0 **并记录截断比例**，不得静默；
- 因果性硬约束：只允许输入当前与历史命令（及必要的自身滞后项），
  **禁止**输入执行侧力矩、执行能量、扰动标签或未来命令。

### 10.4 零/近零 `J_nom` 与不可计算规则

```text
若 J_nom ≥ 1.0 J   →  ε_c = (J_exec − J_nom) / J_nom    依 §10.6 计算
若 J_nom < 1.0 J   →  记为 non_computable
```

- `non_computable` 的记录**单独成列**：计入"全体业务试验"覆盖分母，
  但**不参与** α 的分位数计算；
- **禁止**对分母加任意 `epsilon` 来美化比值；也禁止把 `non_computable` 直接丢弃后宣称覆盖率；
- 阈值 1.0 J 是预登记常数，写进 CAL-009 的协议并散列；
  若实测发现该阈值把过多正常样本判为不可计算（例如 > 20%），
  按 uncertainty_policy 回到本任务重新裁定，**不得**实现中擅自调整。

### 10.5 功率基线：规模、划分与资源预算

| 项 | 值 |
| --- | --- |
| 总条数 | **400**（30 s / 条，clean profile） |
| 划分 | 拟合 240 / 开发 80 / 验证 80（按 20 格 × 4 / 4 / 4，保持命令族与方向覆盖） |
| 划分原则 | 按**整条轨迹或命令簇**分离，不按时间点切分 |
| 与正式数据关系 | 完全独立；pilot 产物不得进入拟合或统计 |
| 资源估算 | 本次实测 8 envs × 8 s ≈ 2 min → 400 条 30 s 约 **5–7 小时**（8 envs）；32 envs 可降到约 1.5–2 小时 |
| 存储 | 每条约 200 Hz × 30 s × 12 关节 × 4 B ≈ 2.9 MB（未压缩），400 条约 1.2 GB；`/pub` 剩余 1.9 T 可容纳 |

**模型候选**（按优先级，先简单后复杂）：

1. **M1 非负线性/二次映射**：特征 `[1, |vx|, |vy|, |wz|, vx², vy², wz², 交叉项]`，
   最小二乘后截断非负。可解释、可复算、优先采用。
2. **M2 非负梯度提升树**：对过渡段非线性更稳，但可解释性下降，且需固定随机种子与版本。
3. **M3 小型 MLP（输出层非负）**：仅在 M1/M2 均未达准入时启用，需额外记录训练配置与散列。

**准入**（在独立验证集 80 条上判定，全部满足才可冻结）：

- 逐轨迹 `|J_nom − J_exec| / J_exec` 的 **90% 分位 ≤ 10%**；
- 非负性 100%（含截断记录）；
- 因果性检查通过（不含执行侧输入）；
- 分组检查：低速、启停、转向子组分别报告误差，不得只报总体。

未达标 → 按 CAL-020 规定记录并暂停，**不得**用 α 吸收模型缺陷。

### 10.6 误差与裕度

```text
ε_c = (J_exec − J_nom) / J_nom        仅 J_nom ≥ 1.0 J
ε_d = max_t ‖p_exec(t) − p_nom(t)‖₂   10 Hz 公共网格（另报 50 Hz 原始最大值作敏感性）
α   = max(0, Q_0.90(ε_c))             90% nearest-rank
δ   = Q_0.90(ε_d)
```

负超支保留原值参与分位；只有落到安全裕度时才取 `max(0, ·)`（§10.2 已保证 `J_exec ≥ 0`）。

### 10.7 上层接口（CAL-022 消费契约）

| 项 | 定义 |
| --- | --- |
| 步成本 | 每个 0.1 s 决策步：`c_t = P̂_nom(·) · 0.1`，单位 **J** |
| 预算扣减 | `z_{t+1} = z_t − (1+α)·c_t`，`z` 单位 J |
| 参考预算 | `z_ref` 必须按 J 重定；**禁止**沿用论文 120/140/190 a.u. |
| history 需求 | 接口预留最近 `K` 条命令（`K ∈ {1,2,4}`，具体值由 CAL-020 的模型输入 schema 决定）；若 `K > 1`，导航观测必须增广 |
| 度量一致性 | `Σ_t c_t` 必须与整条 `J_nom` 在数值容差内一致（CAL-022 验收项） |
| 定长风险 | 30 s 定长下不得退回"时间比"成本；`J_nom` 由模型给出，不是窗口长度 |

### 10.8 与论文的语义差异（不改稿）

论文的成本是 `c(x,u)` 的 a.u. 代理量、决策间隔 0.2 s、`z_ref` 为 120/140/190 a.u.。
本轮实现改为：能量 J、0.1 s、`z_ref` 待重定。
两者的数值**不可互换**，论文不改，差异由本文件与后续交付报告记录。

---

## 11. 采样协议（CAL-009）

协议文件：[`../config/calibration_protocol_v1.0.0.json`](../config/calibration_protocol_v1.0.0.json)
冻结散列：同目录 `calibration_protocol_v1.0.0.json.sha256`
（sha256 `7c4e513f5b5ff5c575a86d90701fcb438967179c78a41c5e38f9b6a3a9d2b095`）

```bash
cd CaMRA/calibration
python3 scripts/validate_protocol.py          # 114 项内部一致性检查
python3 scripts/validate_protocol.py --check-hash   # 比对冻结散列
```

退出码：0 通过 / 2 校验失败 / 3 散列不符 / 4 文件缺失。校验器只用标准库，不需要仿真环境。

协议锁定的关键量（与 §10 一致，不得就地修改）：

| 项 | 值 |
| --- | --- |
| 时钟 | sim 200 Hz(0.005) · policy 50 Hz(0.02) · command 10 Hz(0.1) · pairing 10 Hz · state 50 Hz · energy 200 Hz |
| 窗口 | 1.0 s warmup + 30 s 测量 = 6000 物理步，301 个配对点（含端点） |
| 配额 | baseline 400(240/80/80) · pilot 20 · calibration 800 · heldout 400 · near_obstacle 200，合计 1820 |
| 种子 | 7 段互斥块（4000/4240/4320/5000/6000/7000/8000 起），避开 0、100–109、200–209、300–309 |
| 资源 | 32 envs 实测 484 s/32 条 30 s 窗口；全量约 6–8 h，存储约 5.3 GB |

**变更规则**：任何时钟、命令、扰动、配额、种子、统计口径或容差的改动都必须出新版本号
（`calibration_protocol_v1.1.0.json` 等），旧版本标记 `superseded`，不得就地编辑。

---

## 12. 模块结构与清单格式（CAL-010）

### 12.1 包结构

```text
CaMRA/calibration/
├── pyproject.toml              # camra-calibration 0.1.0，src 布局，仅标准库依赖
├── src/camra_calibration/
│   ├── config.py               # 三级路径解析（env > repro.env > 内置默认）+ preflight
│   ├── protocol.py             # 冻结协议加载、散列校验、quota/seed_block/j_min 访问器
│   ├── manifest.py             # trial 清单 JSONL 的 schema 校验
│   └── cli.py                  # preflight / protocol / validate-manifest
├── tests/                      # 27 项纯 CPU 测试
├── config/calibration_protocol_v1.0.0.json(+.sha256)
├── config/examples/pilot_manifest.example.jsonl
└── scripts/validate_protocol.py   # 协议自身的 114 项一致性检查（独立于包）
```

安装与入口：

```bash
pip install -e CaMRA/calibration
camra-calib preflight                       # 机器配置 9 项检查
camra-calib protocol                        # 展示冻结协议与种子块
camra-calib validate-manifest <清单.jsonl>  # 清单 schema 校验
```

退出码：0 通过 / 2 校验或预检失败 / 3 散列不匹配 / 4 文件缺失。

### 12.2 一条 trial 记录（JSONL 一行）

必备字段：

| 组 | 字段 |
| --- | --- |
| 身份 | `trial_id`（`<split>-<family>-<profile>-<NNNN>`）、`split`、`family`、`profile`、`seed`、`attempt` |
| 版本绑定 | `controller`、`protocol_version`、`protocol_sha256`、`bundle_tree_sha256` |
| 结果 | `outcome`（completed/terminated/truncated）、`steps_executed`（completed 必须 = 6000） |
| 实际配置 | `profile_applied`（friction / added_mass_kg / push_events / obs_noise） |
| 初态 | `initial_state`（px / py / yaw） |
| 产物 | `paths.physics_200hz`、`paths.state_50hz`、`paths.pairing_10hz`、`paths.energy_summary` |
| 能量 | `energy`（unit=J / j_exec / j_nom / j_min=1.0 / non_computable） |

### 12.3 明确拒绝的情形（均由测试固定）

缺字段 · `trial_id` 重复 · `trial_id` 分段与字段不一致 · `split` 非法 · `seed` 不在协议种子块内 ·
绝对路径 · 含 `..` 的目录逃逸 · 后缀不符 · 跨 trial 复用同一产物文件 · `energy.unit` 非 `J` ·
`non_computable` 与 `j_nom < j_min` 不一致 · `protocol_sha256` 与当前冻结协议不符 ·
`outcome=completed` 但 `steps_executed != 6000`。

### 12.4 本任务尚未实现（后续任务负责）

仿真环境构建、命令程序生成、200 Hz 能量计量、配对轨迹采集、α/δ 计算与复算、交付包。
本模块当前只负责"协议 + 配置 + 清单"三件事，不导入 torch / isaaclab。

---

## 13. 名义运动与共享时间轴（CAL-011）

### 13.1 时间轴

`src/camra_calibration/timeline.py` 把协议频率换算成同一 30 s 窗口上的区间数与采样点数，
并断言两端严格对齐。用词固定为 **interval（区间）** 与 **point（点）**，避免 off-by-one 争议：

| 时钟 | 区间数 | 采样点 | 说明 |
| --- | --- | --- | --- |
| 物理 200 Hz | 6000 | 6000 | 步末采样；t=0 初态单独给出 |
| 策略 50 Hz | 1500 | 1501 | |
| 命令 10 Hz | 300 | 301 | 含两端 |
| 配对 10 Hz | 300 | 301 | 含两端 |
| warmup 1 s | 200 | — | 与测量窗口不重叠 |

整数倍关系：physics/policy = 4，policy/command = 5，physics/command = 20。
`Timeline.check()` 返回问题列表，本机实测为 `[]`。窗口不可整除时（例如 sim_dt=0.007）构造即抛错。

**warmup 不会混入正式数据**：warmup 是独立的 200 个物理区间，只用于稳定；
所有记录从测量窗口第 0 个点（t=0）开始，命令序列的第 0 点也是 t=0。

### 13.2 名义运动

`src/camra_calibration/nominal.py`，状态只有平面 `(px, py, yaw)`：

```text
yaw(t+dt) = yaw(t) + wz·dt
p(t+dt)   = p(t) + ∫₀^dt R(yaw(t)+wz·s)·[vx, vy]ᵀ ds
```

分段常量命令下有闭式解（`step_exact`）：`wz=0` 退化为直线，否则用
`(sinθ₁−sinθ₀)/wz`、`(cosθ₁−cosθ₀)/wz` 的解析积分。生产路径**不用数值积分器**。

两套独立对照：

| 对照 | 用途 | 实测 |
| --- | --- | --- |
| `integrate_rk4`（四阶，≥50 子步） | 与闭式解比数值一致性，容差 1e-9 | 四族最大偏差 7.4e-14 ~ 6.3e-13 |
| `integrate_euler`（一阶） | 误差应随步长线性收敛（验证无固定偏置） | 步长减半误差比落在 1.7–2.3 |

**扰动不改变名义轨迹是构造性保证**：`integrate()` 的签名只有命令与初态，
profile、扰动、执行侧量根本不在参数里，因此这不是事后断言。

### 13.3 命令程序

`src/camra_calibration/commands.py`，四个命令族：

| 族 | 内容 |
| --- | --- |
| `straight_shift` | 直行变速（0/±0.1…±1.0）叠加 ±vy 侧移，`wz≡0` |
| `arc_combined` | `vx>0` 配合 ±wz 形成圆弧，含小幅 vy |
| `alternating` | 逐个保持段翻转选定轴的符号（轮换 vx/vy/wz） |
| `start_stop_turn` | 四相循环：静止 → 前进 → 静止 → 原地转向 |

规则：每 1.0 s 一个保持段，段首 0.2 s 线性 ramp（在 t=1.0 s 仍是上一段目标，t=1.2 s 到达新目标），
301 个命令点，全部按命令域裁剪，随机源只有 `random.Random("<seed>:<family>")`。

### 13.4 逐位可复现

`scripts/demo_nominal.py` 输出四族的程序散列、轨迹散列与对账指标。
两个独立进程各跑一次，输出文件 **byte-identical**：

```text
cd4f4c71f1c4e74cd10e05dd6a6f7df0aab3c18f39f55746fba2a555353fd619  c11_nominal_run1.json
cd4f4c71f1c4e74cd10e05dd6a6f7df0aab3c18f39f55746fba2a555353fd619  c11_nominal_run2.json
```

### 13.5 一处协议歧义（已记录，未擅自改语义）

协议把低速度层定义为"`|vx| ≤ 0.3` **或** `|yaw_rate| ≤ 0.3`"，`min_share_of_cell = 0.25`。
按字面的 **或**，`wz≡0` 的直行族会 **100%** 落入低速层，25% 下限失去约束力。

当前处理：命令生成器强制每 3 段中第 3 段满足 **且**（`|vx|≤0.3` 且 `|wz|≤0.3`），
这是对两种读法都成立的更强要求；同时 `Program.low_speed_share(strict=True/False)`
分别给出两种口径的实测占比，把选择留成可审计的观测。实测 strict 占比：
0.45 / 0.29 / 0.65 / 0.68，均 ≥ 0.25。

**待用户确认**：是否把协议的低速度层改为"且"，或保留"或"并另设 `wz≠0` 子层。
若需要改协议，按 §11 的变更规则出 `v1.0.1`，不就地修改 v1.0.0。

---

## 14. 试验清单生成（CAL-012）

### 14.1 模块与产物

`src/camra_calibration/trials.py` + `scripts/build_trial_lists.py`。
清单是 **(split, protocol) 的纯函数**：同一协议重复运行必然得到相同文件
（本机双进程比对 9 个文件全部 IDENTICAL）。产物在
`/pub/data/caohy/logs/camra/trial_lists_v1.0.0/`：

| 文件 | 内容 |
| --- | --- |
| `<split>.jsonl` | 每行一条 trial（trial_id / split / family / profile / seed / attempt / program_sha256 等） |
| `lock.json` | 各 split 条数与整表散列，绑定协议版本与散列 |
| `strata.json` | 逐 split × cell 的分层统计（低速占比、方向/静止/边界覆盖） |

整表散列（写进 `lock.json`，采集前必须核对）：pilot `73be700d…`、
baseline_fit `d3d82f0d…`、baseline_dev `aab21d88…`、baseline_validate `7ef9b10f…`、
calibration `0b62293b…`、heldout `87e7eaf0…`、near_obstacle `0d76083b…`，合计 1820 条。

### 14.2 展开规则

- seed 从该 split 的种子块**顺序**分配（族 → profile → 格内序号），与并行度、
  分批方式无关（测试固定：一次性生成与每 7 条一批的散列一致）；
- 每条 trial 记录其 `program_sha256`，把清单绑定到具体命令序列；
- clean-only 的 split（pilot / baseline_*）只有 4 个族格。

### 14.3 两处协议矛盾（已记录，未擅自改协议）

1. **baseline 的 per_cell**：协议写 `baseline.per_cell = 20`（隐含 20 格），但同一段
   又写 `baseline.profile = "clean only"`——clean-only 只有 4 个族格，4×20=80 ≠ 400。
   实现按总额 400/240/80/80/20 反推格内条数（clean-only 格 = 总额/4）。
2. **低速度层的"或"**（承 §13.5）：字面"或"使 `wz≡0` 的直行族 100% 命中低速层，
   下限失效。此外分层检查还暴露了第三处问题：**只按"每 3 段强制 1 段低速"仍不够**——
   ramp 从高速目标插值进来会吃掉低速点数，`near_obstacle/arc_combined/mass_only`
   格均只有 0.2432 < 0.25。

**当前处理**：命令生成器改为 **segment 0 也强制低速**（与 warmup 后从近静止起步一致），
全部 64 个格（clean-only 4×4 + 五 profile 20×3）的格级低速占比 ≥ 0.282。
`min_share_of_cell` 按**格内平均占比**判定（"share of cell"的自然读法），
逐条最小值只作观测记录。生成器变更使命令程序散列整体更新，
§13.4 的演示输出散列相应变为 `f089d821…`（两次运行仍 byte-identical）。

**待用户确认**（与 §13.5 合并）：是否出协议 v1.0.1 明确（a）低速度层用"且"还是"或"；
（b）`min_share_of_cell` 按格级平均还是逐条最小；（c）baseline clean-only 的格内条数。

---

## 15. 五类物理执行条件（CAL-013）

### 15.1 实现结构

| 模块 | 职责 | 依赖 |
| --- | --- | --- |
| `src/camra_calibration/profiles.py` | 按 trial seed 确定性采样 applied 配置、不变量校验、manifest 记录、knob-diff 矩阵、初态位姿采样 | 纯标准库 |
| `sim/profiles_sim.py` | cfg 级改造 + 逐 env PhysX 直写（摩擦/质量/初态/推扰） | torch / isaaclab |
| `scripts/check_profiles.py` | 无头验证：cfg diff + PhysX 读回（`--profiles` 子集，**每进程只建一个 env**） | AppLauncher |
| `scripts/merge_profile_checks.py` | 合并子集 JSON 为 summary.json | 纯标准库 |

随机流盐：命令 `<seed>:<family>`、profile `<seed>:profile:<id>`、初态 `<seed>:initial`，三者互不重叠。

### 15.2 采样规则（与协议 profiles 段一致）

| profile | friction | added_mass | push | obs_noise |
| --- | --- | --- | --- | --- |
| clean | 1.0 | 0.0 | 无 | 禁用 |
| friction_only | U(0.3,1.25) | 0.0 | 无 | 禁用 |
| mass_only | 1.0 | U(-1,3) kg | 无 | 禁用 |
| push_only | 1.0 | 0.0 | 5-10 s 间隔，±0.5 m/s | 禁用 |
| combined | U(0.3,1.25) | U(-1,3) | 同上 | **训练配置现有值** |

摩擦取 static == dynamic 单一值（restitution 固定 clean 值 0.0，协议未列出的旋钮不变）；
obs_noise 的"训练配置现有值"取自 `velocity_env_cfg.py` PolicyCfg 的 Unoise
（ang_vel ±0.2 / gravity ±0.05 / joint_pos ±0.01 / joint_vel ±1.5），施加时逐项核对。

### 15.3 隐藏随机化全部显式处置（cfg 级）

`configure_env_for_profile` 在 gym.make 前：

- **移除**事件项：`physics_material`、`add_base_mass`（逐 env 直写替代）、
  `base_external_force_torque`（本就是 0）、`push_robot`（时刻表推替代）；
- `reset_base` 位姿区间退化为 0（初态由 trial 指定后直写），速度区间显式全 0；
- `reset_robot_joints` 关节速度区间退化为 0（位置缩放已是 (1,1) no-op）；
- `observations.policy.enable_corruption` 按 profile 开关（combined True，其余 False）。

### 15.4 逐 env 精确写入（PhysX 读回校验）

32 envs 并行时每个 env 属于不同 trial（摩擦/质量/推扰各不相同），
事件项的全局随机范围无法表达，因此全部逐 env 直写：

| 旋钮 | 写入 | 读回 |
| --- | --- | --- |
| 摩擦 | `set_material_properties`（逐 env × 27 shapes，static=dynamic=f，rest=0） | 逐位相等 |
| 质量 | `set_masses`（base 刚体加增量）+ 按质量比缩放默认惯量（同官方实现） | 质量容差 1e-5（float32 1 ULP≈5e-7）、惯量 1e-6 |
| 初态 | `write_root_pose_to_sim`（trial 的 px/py/yaw，z 用默认根高） | 位置逐位、yaw<1e-5 |
| 推扰 | `write_root_velocity_to_sim`（当前速度+[vx,vy,0,0,0,0]，同官方 push_by_setting_velocity） | 读回 delta=0 |

### 15.5 验证结果（profile_checks/）

两个独立进程（clean/friction_only/mass_only/push_only 共用一个 corruption=False env；
combined 单独 corruption=True env），每 profile 2 envs（seed 6000/6001）：

- **PhysX 读回 20/20 项全过**（摩擦 10、质量 10、初态 10、推扰 2），`overall_ok=True`；
- **cfg 级 diff vs clean**：clean/friction_only/mass_only/push_only 完全一致（diff 为空），
  仅 combined 差 `obs_corruption_enabled=True` + 噪声值——证明 cfg 层只有指定旋钮变化；
- **分组标签与实际配置一致**：`AppliedProfile.check_invariants()` 校验 knobs_changed
  集合与协议 PROFILE_KNOBS 逐 profile 相等（单测固定），applied 记录含摩擦/质量/
  push 时刻与幅度/噪声，可写入 manifest 的 `profile_applied` 字段。

### 15.6 踩坑记录（对 CAL-014/018 采集器直接有效）

1. **同一进程 `env.close()` 后再 `gym.make` 会挂死**（108% CPU 自旋，两次复现，
   与 corruption 无关）——采集器每进程只建一个 env；多 profile 校验分进程跑。
2. **`sim_app.close()` 会强制退出进程**：所有落盘/打印必须在 close 之前完成，
   否则退出码 0 但证据全丢（本次实际发生过）。
3. `get_root_transforms()` 的四元数是 **(qx,qy,qz,qw)**，而 `robot.data.root_quat_w`
   是 (w,x,y,z)——yaw 恢复公式必须按前者写。
4. PhysX `set_masses/set_material_properties` 收 **CPU** int64 索引，
   `write_root_pose/velocity_to_sim` 收 **GPU** 索引。
5. 质量写入后 PhysX 会重算惯量（float32 舍入 ~1e-9），读回比较必须用容差。
6. `pkill -9` 仿真进程可能把 CUDA 状态打坏（"Skipping NVIDIA GPU due CUDA being
   in bad state"）——停仿真优先用 tmux 会话与普通 kill。

---

## 16. 200 Hz 能量计量与 50 Hz 状态记录（CAL-014）

### 16.1 实现

| 模块 | 内容 | 依赖 |
| --- | --- | --- |
| `src/camra_calibration/energy.py` | `EnergyMeter`（在线逐物理步累计）、`integrate_series`（独立离线复算）、`steps_for`（非整数窗口直接报错）、`reference_naive_sum_then_clip`（**仅作反例对照**） | 标准库 + numpy |
| `sim/recorder.py` | `TrialRecorder`：200 Hz 物理记录（tau / qdot / 根位姿 / 根速度）+ 50 Hz 状态（动作 / 命令 / 位姿 / 关节量）+ 每 env 一个计量器 | torch |
| `scripts/check_meter.py` | 单回合 31 s（1 s warmup + 30 s 窗口）驱动，`--mode on|off` | AppLauncher |
| `scripts/verify_meter_runs.py` | on/off 逐字段比对 + 离线复算 + 完整度（不启动仿真） | torch |

### 16.2 三条硬约束的落实

1. **逐物理步**：`TrialRecorder.after_physics_step()` 挂在 `sim.step` 包装上，每个子步读一次
   `applied_torque` 与该子步 `robot.update(sim_dt)` 后的 `joint_vel`（§9.3 的时间戳门控），
   逐子步累加；不存在"每 4 子步抽 1 点"。
2. **逐关节截断**：`max(τ_j·q̇_j, 0)` 先按关节截断再求和；反例函数
   `reference_naive_sum_then_clip` 在测试里证明两者在关节间正负功场景下结果不同
   （契约 0.01 J vs 反例 0.0 J）。
3. **只观察**：模块不调用任何 `set_*`/`write_*`；是否扰动动力学由 on/off 双跑实测（§16.4）。

warmup 与窗口：`begin_warmup()` 后的物理步只计数（200 步），`begin_window()` 之后才开始
累计，因此 warmup 不可能混入能量；`begin_window()` 重复调用直接报错。

### 16.3 独立参考结果（15 项 pytest）

| 场景 | 参考 | 结果 |
| --- | --- | --- |
| 常功率 100 步 × 12 关节 × 2 W | 解析 12 J | 一致（rel 1e-15） |
| 正负交替 | 恰为全正情形的一半 | 一致 |
| 关节间抵消（+2 W 与 −2 W 同刻） | 契约 0.01 J / 反例 0.0 J | 明确不等，反例被记录为禁用 |
| 全负功 | 0 | 一致 |
| 非整数窗口 2.5 s @ dt=0.006 | 必须报错；417 步 → 2.502 s | `steps_for` 抛错；计量器按 417×0.006 精确累计 |
| 单位换算 | mN·m→N·m、deg/s→rad/s | 与 J 口径差 <1e-12 |
| 随机 500 步 | 在线 vs 离线 | 一致（rel 1e-12） |

### 16.4 单回合实测（meter_check/）

clean profile、trial_seed 6000、straight_shift、2 envs、episode 余量 0.2 s：

| 项 | 结果 |
| --- | --- |
| 单回合结构 | 200 warmup 步 + **6000 窗口步** + 1500 控制步，outcome=completed |
| 能量 | env0 **507.1266 J**、env1 **493.3150 J**（30 s，单位 J） |
| 离线复算 | 相对误差 **2.9e-15 / 9.2e-16**（协议容差 1e-5） |
| 测量开关影响 | 8 个 50 Hz 字段（动作/命令/位姿/四元数/根线速度/根角速度/关节位置/关节速度）**逐位相同**，max abs diff = 0 |
| 完整度自检 | `EnergyMeter.verify_complete()` 空；逐步求和与逐关节求和均等于总量 |

第一版把 `episode_length_s` 设为正好 31.0 s，窗口最后一步与 `time_out` 撞在同一控制步、
被记成 `truncated`。这不是业务失败，而是时限设置问题：现加 0.2 s 余量并在启动前断言
`max_episode_length > warmup_ctrl + window_ctrl`，落盘记录 `episode_length_s=31.2`。

### 16.5 产物

`/pub/data/caohy/logs/camra/meter_check/`：
`meter_on.npz`（tau/qdot/根状态 200 Hz + 全部 50 Hz 字段，2.4 MB）、
`meter_off.npz`（仅 50 Hz）、`meter_on.json` / `meter_off.json`（元数据与能量摘要）、
`verify_report.json`（逐字段 diff、离线复算、完整度、VERDICT pass）。

---

## 17. 配对轨迹采集器（CAL-015）

### 17.1 结构

| 文件 | 职责 |
| --- | --- |
| `sim/batch_program.py` | `TableProgram`：每 env 一张 10 Hz 命令表，`at(t)` 返回 `(num_envs,3)`；warmup 期间强制零命令 |
| `scripts/collect_trials.py` | 采集器：读清单 → 按 bucket 分批 → 逐 env 施加 profile/初态/命令 → 1 s warmup + 30 s 窗口 → 逐 trial 原子落盘 |
| `scripts/check_collection.py` | 离线核对：覆盖/重复/缺失、形状与口径、能量复算、元数据自洽 |

一个进程只建一次 env（§15.6 的二次 `gym.make` 挂死），按 **bucket** 分进程：
`combined`（obs corruption 开）与其余四个（关）不能同进程，因为该开关是 cfg 级。

### 17.2 关键机制

- **每 env 一条独立 trial**：命令表、摩擦/质量、初态位姿、push 时刻全部按该 env 的
  trial seed 采样；`TableProgram` 让同一批里的每个 env 跑不同命令族。
- **warmup 边界**：先跑 50 个控制步（200 物理步）零命令，`begin_window()` 之后才开始
  记录；实测 `warmup_steps == 200` 否则直接报错。
- **时序上限**：`episode_length_s = 1 + 30 + 0.2`，并在建 env 后断言
  `max_episode_length > warmup_ctrl + window_ctrl`（训练默认 20 s 会截断）。
- **终止不拼轨迹**：任一 env 触发 termination 立即停止整批（IsaacLab 会在下一步自动
  复位，继续跑就会把新轨迹写进同一数组）；该 env 记 `terminated`，兄弟 env 记
  `interrupted`，两者都落盘、都计入覆盖分母，且 meta 注明"未外推补足 30 s"。
- **原子保存**：每个文件先写 `.tmp` 再 `os.replace`，`DONE` 最后写；续采只看 `DONE`，
  已存在但没有 `DONE` 的目录改用 `<trial_id>__a2`，绝不覆盖旧分片。
- **命令/名义一致**：同一条 `build_program(family, seed)` 既生成下发命令表，也用于
  `nominal.integrate` 出 301 点名义轨迹，因此"同一实际指令"是构造性保证。

### 17.3 产物口径

每条 trial 一个目录（`trials/<split>/<trial_id>/`）：

| 文件 | 内容 | 形状 |
| --- | --- | --- |
| `physics_200hz.npz` | tau / qdot / 根位姿 / 根速度 / sim_dt | (6000, 12) 等 |
| `state_50hz.npz` | action / cmd / 根位姿 / 根速度 / 关节位置速度 | (1500, …) |
| `pairing_10hz.npz` | t / exec_pos / nominal_pos / nominal_yaw / deviation / cmd | (301, …) |
| `energy.json` | j_exec / per_joint_j / unit / j_nom 状态 | — |
| `meta.json` | 身份、版本、outcome、步数、profile_applied、初始位姿、cfg 报告 | — |
| `DONE` | 完成标记（原子性依据） | — |

配对点 = **t=0 初态 + 每 0.1 s 的 300 个点**（取 50 Hz 第 5k−1 条），共 301 个，
与 `Timeline.pairing_points` 一致；能量按 D-ENERGY-001 逐物理步累计。

`energy.j_nom` 目前写 `null` + `j_nom_status="not_available_pending_power_model"`：
名义功率模型属于 CAL-019/020，采集阶段不能凭空造数。`records.jsonl` 是采集索引，
最终 manifest（含 j_nom）在各 split 采集完成、功率模型就绪后统一装配。

### 17.4 pilot 实测（20 条，split=pilot，bucket=off）

| 项 | 结果 |
| --- | --- |
| 覆盖 | 20/20 落盘，records 20 条，无缺失/重复/清单外 |
| 结果分布 | completed 12、interrupted 7、terminated 1（真实摔倒，未补足、未拼接） |
| 形状 | 完整 trial：6000 物理步 / 1500 控制步 / 301 配对点（0→30.0 s，dt=0.1） |
| 能量 | j_exec 156.55–679.56 J；离线复算最大相对误差 **8.1e-10** |
| 一致性 | 逐关节求和 = j_exec（515.808069 J）；50 Hz×4 = 200 Hz 步数 |
| 判定 | `check_collection.py` **VERDICT pass** |

### 17.5 吞吐实测与并行度结论（重要）

同一批 8 envs × 30 s 实测 wall **421–484 s**；微基准分解每物理子步：
`sim.step` **63.4 ms**、`robot.update` 4.5 ms、数据读取 4.5 ms、能量运算 2.3 ms。
`sim.step` 与环境数几乎无关（2 envs 时也是 63 ms），说明当前瓶颈是**GPU 争用**
（同时 23 个用户、load 130–160），不是我们的记录代码。

因此吞吐取决于每步摊到多少 env：8 envs ≈ 60 s/trial，**32 envs ≈ 13–15 s/trial**。
正式采集（CAL-018 pilot 与后续 split）应使用 `--num_envs 32`；
按该口径 1820 条约 6–8 h，与协议 `resource_budget` 的估算一致。

第一版记录器逐子步 `.tolist()` + 6 次 GPU→CPU 拷贝，慢到 480 s/8 条；
现改为 **GPU 预分配缓冲 + 设备端 float64 累加**，只在 `finish()` 一次性搬回，
记录开销降到每子步 <10 ms。

---

## 18. 配对、失败分类与误差独立复算（CAL-016）

### 18.1 口径

| 项 | 定义 |
| --- | --- |
| 配对网格 | 10 Hz 公共网格 301 点（t=0 与 t=30 s 含端点），位置按时间**线性配对** |
| 对齐 | 位置**不做任何相位对齐**（不平移、不 DTW）；yaw 只作运动学/诊断 |
| 正式 ε | 仅 `outcome=completed` 且数据自洽（网格/形状/步数全部通过）才产出 |
| 前缀诊断 | terminated / interrupted / truncated 只保留前缀诊断，`official=false`，计入失败侧 |
| ε_c | `(J_exec − J_nom)/J_nom`；`J_nom` 缺失记 `j_nom_missing`；`J_nom < J_min` 记 `non_computable`（单列、不进 α 分位）；负超支保留原值 |
| ε_d | `max_t ‖p_exec − p_nom‖₂`；50 Hz 敏感性 = 把名义轨迹线性插值到 50 Hz 时刻后的最大偏差 |
| 失败分类 | `business_failure`（stalled/low_speed/terminated/interrupted/truncated）、`data_invalid`（nan/time_jump/non_monotonic/zero_window/shape_mismatch/missing_shard/duplicate_trial/wrong_source）、`infra_retry`（采集日志判定） |

实现：`src/camra_calibration/pairing.py`（纯标准库 + numpy）与 `scripts/recompute_errors.py`。

### 18.2 一个真实数据缺陷（已修）

首版采集把 `pairing_10hz.npz` 的 `t` 写成 **float32**。0.1 在 float32 下不可精确表示，
300 段累积后 `max|Δ−0.1| = 1.53e-6`，超过协议 `time_sync_abs_s = 1e-6`，
导致 20 条 trial 全部被判 `time_jump`。

处置：① 采集器改为写 **float64** `t`（新数据精确到 1e-16）；
② `check_time_grid` 的容差按**存储精度**取——float32 网格 1e-5、float64 网格 1e-6，
用于兼容早期产物且在报告里可见。这条修改同时修正了 `aggregate` 把 `None` 当数值
转出 NaN 的问题。

### 18.3 pilot 复算结果（20 条，clean）

| 项 | 结果 |
| --- | --- |
| 分母 | all 20 / observed 20 / **complete 12** / business_failed 8 / **data_invalid 0** |
| ε_d（正式，12 条） | 0.5217 – 4.8632 m，均值 1.4609 |
| ε_d（前缀诊断，8 条） | 0.7401 – 3.7012 m，均值 1.6037（`official=false`） |
| ε_d(50 Hz 敏感性) | 0.5217 – 4.8632 m，均值 1.4609（与 10 Hz 最大差 4.4e-5 m） |
| ε_c | **0 条可用**：`J_nom` 尚未就绪（功率模型属 CAL-019/020），按 `j_nom_missing` 记录，未造数 |
| 低速占比（命令侧） | 0.0166 / 0.2592 / 0.6013（min/mean/max） |
| 失败样例 | `pilot-straight_shift-clean-0000`：outcome=terminated、prefix_only、ε_d=1.731 m（前 259 个配对点）、error_class=business_failure |

排除计数与原因在 `errors_report.json` 的 `exclusions` 字段逐条给出；
任何未观测到的 trial 记 `missing_shard`，重复 attempt 记 `duplicate_dirs`。

---

## 19. CPU 数值、划分与恢复逻辑测试（CAL-017）

### 19.1 覆盖

| 文件 | 项数 | 覆盖 |
| --- | --- | --- |
| `tests/test_pairing.py` | 24 | ε_c 精确值与负超支、j_nom 缺失/近零分母、ε_d 取 max 非 mean、网格点插值精确性与外推拒绝、网格起点偏移/跳变/非单调/NaN/单点、位置形状与 NaN、完整 vs 前缀诊断、NaN→data_invalid（不误判业务失败）、completed 但步数不足→data_invalid、unknown outcome→wrong_source、分母与排除计数、None 跳过、低速"且"语义、50 Hz 敏感性一致、重复 trial 检测 |
| `tests/test_store.py` | 13 | 批次切片保序、空/非法 n_env、**不同批大小 records 完全一致**、续采幂等（含跳过计数）、allow_overwrite、首次 attempt=1、已存在无 DONE→`__a2/__a3` 且原目录不动、崩溃后恢复新建 attempt、DONE 在后缀目录也算完成、原子写不留 `.tmp`、records 追加不截断、跨 split 种子块互斥且配额精确、批大小不改变整表散列 |
| 既有 | 101 | 协议/清单/时间轴/名义/命令/profile/能量等 |

合计 **138 项 pytest 全绿**（`python -m pytest CaMRA/calibration/tests -q`）。

### 19.2 故意构造的错误路径（都必须失败而不是静默通过）

缺分片 → `missing_shard`；损坏 npz → `missing_shard`；重复 trial → `duplicate_dirs` +
`duplicate_trial`；乱时钟（起点偏移/非单调/间隔跳变）→ `time_jump`/`non_monotonic`；
NaN/Inf → `nan`；业务失败（terminated/interrupted）→ `business_failure` 且 `official=false`；
未知 outcome → `wrong_source`。上述每一条都有对应断言。

### 19.3 关于 fixture 的边界

`test_pairing.py` / `test_store.py` 的解析场与合成误差一律标注 **test_fixture**，
只用于验证口径与错误分支；**不得**当作仿真证据。真实数据结论只来自
`trials/<split>/` 下的采集产物与 `errors_report.json`。

---

## 20. 协议 v1.0.1 与 pilot 五条件闭环（CAL-018）

### 20.1 v1.0.1 的四处变更（用户 2026-11-04 裁定）

| # | 变更 | 依据 |
| --- | --- | --- |
| 1 | pilot 由 clean-only 改为**覆盖五类执行条件**（4 族 × 5 profile = 20 格，每格 1 条），种子块 5000–5019 → **5100–5119**，旧块列入保留 | CAL-018 要求"四族×五条件"，与 v1.0.0 的 clean-only 冲突 |
| 2 | 低速度层由"或"改为**"且"**（`|vx| ≤ 0.3` 且 `|wz| ≤ 0.3`） | 字面"或"使 `wz≡0` 的直行族 100% 命中，下限失效 |
| 3 | `min_share_of_cell` 明确按**格级平均**判定，逐条最小值只作观测 | 原表述未定层级 |
| 4 | baseline 的 clean-only 格内条数显式写明：cells=4、per_cell=100（fit 60 / dev 20 / validate 20） | 原 `per_cell=20` 隐含 20 格，与总额 400 矛盾 |

v1.0.1 冻结散列：`16aa5ebeb24f4d0d2d7810d646b8e36f7d783d5e93efdedb5eef88432a71f2ac`
（`validate_protocol.py` **119/119** 通过，含新增的 pilot 覆盖与低速层检查）。
v1.0.0 保留为 `superseded`，未就地修改。

### 20.2 为保证 v1.0.1 低速层成立而做的两处生成器修正

pilot 每格只有 1 条，"格级平均"退化为逐条判定，因此**每条**程序的低速占比都必须 ≥ 0.25：

1. `alternating` 只翻转一个轴，其余轴可能仍是高速 → 强制低速段改为**全轴幅值钳制**
   （`|vx| ≤ 0.3` 且 `|wz| ≤ 0.3`）；此前 seed 5103 仅 0.2027。
2. 低速段的 ramp 也**从低速包络内的起点出发**（而非从上一段的高速目标插值），
   使整段 10 个点都落在低速层；代价是段边界可能出现高速→0.3 的阶跃，这正是
   "减速停车"类指令的真实形态。

修正后 ~1300 条程序（4 族 × 4 个 seed 区间）的最小逐条占比 **0.402**（此前 0.2027）。

### 20.3 pilot 实测（v1.0.1，20 条，32 envs）

| 项 | 结果 |
| --- | --- |
| 结构 | 4 族 × 5 profile = **20 格，每格 1 条**，seed 5100–5119 |
| 结果 | **20/20 completed**（无终止、无中断），每批 6000 物理步 / 1500 控制步 / 301 配对点 |
| ε_d | 0.5050 – 2.3645 m，均值 **1.2503**（20 条全部正式，无前缀诊断） |
| 能量（按条件） | clean 430.4 / friction_only 411.1 / mass_only 497.1 / push_only 410.7 / combined 476.8 J（均值） |
| 五条件实际值 | friction_only f=0.6385；mass_only Δm=0.680 kg；push_only 4 次推扰；combined f=1.1224 + 0.669 kg + 4 次 + 训练噪声 |
| 检查 | `check_collection.py` VERDICT pass；`recompute_errors.py` 退出码 0 |
| 吞吐 | 16 条/批 488.8 s、4 条/批约 480 s → **固定成本由 sim.step 决定**（见 §17.5），1820 条约 7–8 h |

### 20.4 故障注入（工具 sandbox，不触碰正式数据）

`scripts/fault_injection_check.py` 从一条真实完成 trial 复制副本并逐项注入故障，9/9 命中预期：

| 场景 | 预期 | 实际 |
| --- | --- | --- |
| baseline（真实 trial） | ok / official | ok |
| 目录不存在 | data_invalid:missing_shard | 一致 |
| npz 截断 | data_invalid:missing_shard | 一致 |
| exec_pos 注入 NaN | data_invalid:nan | 一致 |
| 时间轴 +0.01 s | data_invalid:time_jump | 一致 |
| 同名目录重复 | `{trialX: 2}` | 一致 |
| outcome 改写为 terminated | business_failure、official=False | 一致 |
| 删除 DONE | 重新入列 1 条 | 一致 |

### 20.5 恢复端到端（沙箱副本）

把 `trials/pilot` 复制到 `trials_resume_test/`，删掉
`pilot-alternating-mass_only-0000/DONE` 后原样重跑采集：

```text
selected=16  skipped_done=15  processed=1  outcome=completed  6000 步  wall 472 s
→ 新建 pilot-alternating-mass_only-0000__a2，原目录（含全部产物）保持不动
```

即：恢复只补未完成 trial、保留 attempt、绝不覆盖旧分片，且不重采已完成数据。

### 20.6 v1.0.0 产物的处置

旧 pilot（20 条 clean-only）与旧清单已归档到
`/pub/data/caohy/logs/camra/trials_superseded_v1.0.0/pilot` 与
`trial_lists_v1.0.0_superseded/`，**不进入**任何正式统计或功率基线拟合
（协议 split.pilot.rule 明确 pilot 产物不得进基线拟合/正式统计）。

---

## 21. CAL-020 名义功率模型：预登记与实现（数据无关部分）

CAL-020 在形式上依赖 CAL-019 的产物（拟合必须用 240 条 `baseline_fit`），但
**模型候选、选型规则、验收口径与全部代码不依赖任何基线数值**。本节记录在
采集进行中先完成的部分：预登记先于数据冻结，反而比"看完数据再定规则"更严格。

### 21.1 预登记（`config/power_model_prereg_v1.json`，sha256 `fb00b366…`）

| 项 | 内容 |
| --- | --- |
| 冻结时刻 | 2026-11-04T12:35+08:00（此时 `baseline_fit` 采到 224/240，`dev`/`validate` 尚未开始） |
| 目标量 | `J_nom = Σ_{i=0..299} P̂_nom(u_i, …, u_{i−K}) · 0.1`，单位 J，30 s 窗口 |
| 允许输入 | 当前命令 `u_i=[vx,vy,wz]` 与过去 K 条命令（10 Hz 命令点） |
| 禁止输入 | 执行侧力矩/关节速度/功率、`J_exec`、扰动标签、profile 名、质量/摩擦参数、未来命令 |
| 输入源 | `pairing_10hz.npz:cmd`（301×3 float32），且必须与由 `(family, seed)` 重建的 `Program` 逐点一致（atol 1e-5） |
| 三重绑定 | `Program.sha256()` == 清单 `program_sha256` == `meta.json.program_sha256`；任一不符即排除 |

候选（§10.5 的 M1 展开，先简单后复杂）：

| 名称 | 特征块 | K | 参数量 |
| --- | --- | --- | --- |
| `C1_lin_k0` | `1,|vx|,|vy|,|wz|` | 0 | 4 |
| `C2_quad_k0` | 加 `vx²,vy²,wz²,|vx||wz|,|vy||wz|,|vx||vy|` | 0 | 10 |
| `C3_quad_k1` | 二次 + 1 条历史块 | 1 | 20 |
| `C4_lin_k2` | 线性 + 2 条历史块 | 2 | 12 |

选型规则（只用 `baseline_dev`）：取 dev 逐轨迹相对误差 90% nearest-rank 分位最小者；
若其它候选落在最佳值 **+0.5 个百分点**以内，则按 `n_params` 少 → `history_k` 小 → 名称升序
取更简单者。`validate` 只在**冻结（权重与散列落盘）之后**评估一次，且不得回头改选型。

排除原因预登记为 8 类（`missing_dir / not_done / outcome_not_completed / steps_mismatch /
program_mismatch / cmd_mismatch / j_exec_invalid / pairing_missing`），脚本对未知原因直接断言失败。

### 21.2 为什么非负与因果是构造性的

- 特征只由 `|vx|,|vy|,|wz|` 及其两两乘积构成 → 恒 ≥ 0；权重由 NNLS 求解 → 恒 ≥ 0；
  因此 `P̂ ≥ 0`、`J_nom ≥ 0` **不需要截断**。`truncation_share` 仍逐条计算并上报（实测 0）。
- 区间 `i` 的特征只取 `u_i` 与 `u_{max(0,i−l)}`，越界钳到首点；改动 `u_{>i}` 不可能改变
  第 `≤ i` 个区间的输出（实测前缀差 0.0），这条在拟合、评估、复算三处都跑。

### 21.3 实现与测试

| 文件 | 作用 |
| --- | --- |
| `src/camra_calibration/power.py` | 特征/设计矩阵、NNLS 拟合、`PowerModel`（预测/域检查/截距份额/序列化+散列）、选型、分组评估、准入判定 |
| `scripts/fit_power_model.py` | 装配 → 拟合 → 选型 → 冻结 → 评估 validate → 报告；`--check-split` 只做装配检查不拟合 |
| `tests/test_power.py` | 25 项：特征非负/因果前缀不变/历史钳位、精确权重回收（1e-9）、确定性、序列化往返与篡改检测、域检查、选型三级 tie-break、分组与非可算、准入阈值 |
| `config/power_model_prereg_v1.json(+.sha256)` | 预登记，脚本强制校验侧车散列 |

全量 pytest：**163 passed**（原 138 + 新增 25）。

### 21.4 装配路径实测（pilot，仅连通性，不拟合）

`--check-split pilot`：20/20 可用、0 排除、`cmd` 长度均为 301、`J_exec` 150.53–617.75 J。
证明三重绑定（清单/meta/重建 Program）与 `cmd` 逐点比对在真实产物上成立。

### 21.5 待完成（依赖 CAL-019 采完）

1. `baseline_fit` 240 / `dev` 80 / `validate` 80 全部 completed 后运行
   `scripts/fit_power_model.py`（不加 `--check-split`），产出 `config/power_model_v1.json` 与报告；
2. 按 §10.5 判定：dev/validate 的 q90 ≤ 10%、非负 100%、截断 0、可复算逐位一致；
3. 未达标则按预登记 `on_failure` 记录并暂停——不得先用 α 吸收，也不得擅自装 sklearn 上 M2。

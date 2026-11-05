# HiRA 接入与独立校准 — 实施交接笔记

本文件是本轮（`hira_go2_18500`）的唯一交接入口。v2 的历史证据留在
`CaMRA/calibration/docs/implementation_handoff.md`（§1–§32），**不在此重写**。

- 计划：`doc/plan/20261105_camra_hira_policy_calibration.json`（40 项，0/40 起步）
- 实验身份：`experiment_id = hira_go2_18500`，候选控制器 `camra_hira_go2_18500_lab_v1`（HIRA-020 才冻结）
- 本轮边界：只复用 HiRA 的**底层 actor**；不复用其高层导航网络；不声称原 HiRA 实现了 CaMRA 的预算双评价

---

## 1. 基线（HIRA-001，2026-11-05 15:32 CST）

### 1.1 版本

| 对象 | 值 |
|---|---|
| 主仓库 | `/home/caohy/repositories/Paper`，`main`，HEAD **13fd248**（"docs: plan HiRA controller integration and calibration"） |
| 上游 | `origin/main`，与 HEAD 一致 |
| 工作树 | 0 个未提交变更 |
| HiRA 子模块 | **6a05af55**（`heads/master`），自身 0 个未提交变更，**本轮未拉取新提交** |
| 旧校准计划 | `doc/plan/20261103_camra_calibration_implementation.json`，**32/32 完成**，13 条决策记录，无未完成项 |

### 1.2 设备与环境

- 主机 `tiger10.ios.ac.cn`，账号 `caohy`，时区 CST(+0800)
- GPU：3× A100-PCIE-40GB，驱动 560.35.03
- 环境 `env_isaaclab`：`/pub/data/caohy/miniconda/envs/env_isaaclab/bin/python`，
  torch 2.7.0+cu126 / CUDA 12.6 / isaaclab 0.47.2 / numpy 1.26.0（与 v2 采集期一致）

**显存归属（实测）**：

| GPU | 占用 | 归属 |
|---|---|---|
| 0 | 1922 MiB | 他人：`zhengyangz` 两个 katago（952 + 954 MiB） |
| 1 | 2038 MiB | 他人：`caozp` 的 python |
| 2 | 4 MiB | **空闲**（我方 2.8 GiB 残留已回收） |

### 1.3 tmux 与目录

- 本轮会话：**`camra-hira-calibration`**，窗口 `onboarding / env / adapter / integration / pilot / baseline / calibration / heldout / near / audit`
- 旧会话 `camra-env`（46 窗口）保留不动；其最后一轮 v2 任务已结束，无 `collect_trials` 或 `env_isaaclab` 进程存活
- 日志根（新，隔离）：`/pub/data/caohy/logs/camra_hira/{env_setup,integration,trials}`
- 实验根：`CaMRA/calibration/experiments/hira_go2_18500/{config,reports,docs,tests}`
- 与 v2 的 `/pub/data/caohy/logs/camra` 完全不交叉；本轮未写入、未移动、未删除任何 v2 文件

### 1.4 权重身份（只读核对，非 HIRA-011 的正式提取）

| 文件 | sha256 | 与计划登记 |
|---|---|---|
| `HiRA/logs/rough_go2/Sep08_11-57-26_/model_18500.pt`（4.4 MB） | `d30e5212…03d2e8` | **一致** |
| `HiRA/logs/rough_go2/exported/policies/policy_1.pt` | `6d6f4d36…d6f9c6` | 一致 |
| `HiRA/legged_gym_go2/deploy/pre_train/go2/policy_1.pt` | `0069ce66…67462f` | 一致 |

三者字节不同。身份以**从指定 checkpoint 重新提取 actor** 为准（HIRA-011），不以文件名互换。

### 1.5 资源实测

- `/pub`：29 T 总量，已用 27 T，剩 **1.3 T（96%）**
- v2 `trials` 占 **68 G**，`logs/camra` 合计 69 G
- 结论：HiRA 再来一整套 1820 条约 70 G，空间够；但吞吐必须在本机重测，不能沿用 v2 的估算

---

## 2. 风险登记（接入前已知，逐条挂在具体任务上）

> 这些是 HIRA-001 静态阅读得到的**待验证**风险，不是结论。每条都写清证据位置、影响与验证方法；
> 验证通过后要在对应任务记录里改为 closed，并附证据。

### R-01 `heading_command = True` 是继承来的，Go2 底层没关

- **证据**：`base/legged_robot_config.py:44` 默认 `heading_command = True`；`go2/go2_config.py` 只覆盖了 `env/init_state/control/asset/terrain`，**没有 commands 覆盖**
- **影响**：外部写入的 yaw rate 命令，可能在命令重采样时被 heading 误差反算覆盖 —— 策略看到的命令和外部写的命令不是同一个
- **验证**：HIRA-016。固定 `commands[:, :3]` 跨过至少一个重采样周期，记录策略实际消费的命令并与写入值逐项比对
- **状态**：open

### R-02 PD 在每个物理子步重算，不是每策略步算一次

- **证据**：`base/legged_robot.py:67-74` —— `for _ in range(decimation): _compute_torques(actions) → set_dof_actuation_force_tensor → simulate`
- **影响**：若 Lab 适配层“每策略步算一次力矩、重复施加 4 次”，轨迹与 200 Hz 能量积分都会不等价
- **验证**：HIRA-014 / HIRA-019。同状态同动作，对照一个策略步内 4 个子步的力矩序列
- **状态**：open

### R-03 源 reset 随机化关节角，且从 1 m 落下

- **证据**：`base/legged_robot.py:_reset_dofs` 用 `default_dof_pos * rand(0.5, 1.5)`；`base/legged_robot_config.py:52` `init_state.pos = [0,0,1]`
- **影响**：本轮采集要求固定起点、同 seed 可复现；照搬训练 reset 会让每条 trial 都不同
- **验证**：HIRA-016。同 seed 两次运行的首帧关节角与根位姿必须一致，或差异被显式记录为有意设计
- **状态**：open

### R-04 `action_scale` 有两个来源（基类 0.5 / Go2 覆盖 0.25）

- **证据**：`base/legged_robot_config.py:66` 为 0.5；`go2/go2_config.py` 的 `control` 段为 **0.25**，子类生效
- **影响**：差一倍会彻底改变行为，但机器人“照样能走”，冒烟测试抓不出来
- **验证**：HIRA-008 / HIRA-014。适配层与测试都显式断言解析值为 0.25，并用源 `_compute_torques` 做 oracle 对照
- **状态**：open

### R-05 共享机器：他人占显存、负载 172、磁盘 96%

- **证据**：见 §1.2 与 §1.5
- **影响**：GPU0 在 katago 忙时会让 Isaac 启动卡死（v2 期实测）；批大小与吞吐需重测；新增 70 G 数据需在预算里登记
- **验证**：HIRA-006 / HIRA-009。每次运行指定空闲 GPU，实测首批吞吐后再登记预算
- **状态**：open

### R-06 重试口径已变：成功重试不得替换业务失败

- **证据**：计划 `repeated_calibration_contract.retry`；校准 README 2026-11-05 改写，明确 v2 的一次性补采不推广
- **影响**：v2 近障那格 8/10 → 9/10 只是历史特例；HiRA 的失败 trial 必须留在主分母里，attempt 全保留
- **验证**：HIRA-023 / HIRA-024。采集器测试：terminated 计为失败，且重试不从分母里顶掉原 trial
- **状态**：open

---

## 3. 本轮已做的动作（HIRA-001）

1. 只读核查主仓库、子模块、旧计划与环境（未修改任何旧产物）
2. 建立 tmux 会话 `camra-hira-calibration`（按计划的十个窗口名）
3. 建立实验根与隔离日志根
4. 回收我方残留：`47_diag_near.py`（11:21 启动、跑了 4 h 10 min、占 GPU 2 的 2.8 GiB），
   不属于任何登记任务；GPU 2 现已空闲
5. 明确**未做**：未启动任何实验、未训练、未拉 HiRA 新提交、未改 v2 证据、未动稿件

## 4. 下一步

- **HIRA-002**：理解论文方法与本轮重复校准的科学问题（只读，产出职责与输入/输出说明）
- 之后 HIRA-003/004 分别追踪现有校准实现与 HiRA 执行语义，HIRA-005 汇总可核查的接手地图
- 数值类门禁（HIRA-009 起）**必须先登记判据再测**；任何一处接入不等价，后续采集不得开始

---

## 5. 方法理解与三个不同对象（HIRA-002，2026-11-05）

> 全部为只读学习结果。论文正文、数字、公式、图片均未改动，也未在本文件同步任何论文表述。

### 5.1 问题定义（只读理解）

上层要解的是**最小成本到达避障（MCRA）**：在预算 `z` 内到达目标、避开危险集 `F`、并让累计成本尽量小。
CaMRA 的做法是把"底层能不能稳住"和"上层怎么决策"分开，再用两个经验裕度把底层的不完美折算进上层的成本与安全约束。

两阶段：

1. **冻底层 + 校准裕度**：训练并冻结速度跟踪器，采集"名义—执行"配对轨迹，估出 α 与 δ；
2. **上层学习与部署**：把 `(1+α)c` 作为成本、把 `F ⊕ disc(δ)` 作为危险集，训练导航策略并选初始预算。

### 5.2 四个组件的输入、输出与职责

| 组件 | 输入 | 输出 | 职责 | 不是什么 |
|---|---|---|---|---|
| **冻结控制器**（v2 或 HiRA actor） | 机身速度命令 `[vx, vy, wz]` + 本体观测（角速度、重力投影、关节角/角速度、上一动作） | 12 维关节位置偏移 → PD → 每子步力矩 | 把速度命令变成可执行的关节动作，维持平衡与步态 | 不负责避障、不做预算决策 |
| **名义功率模型** `P̂_nom` | 命令序列 `u`（可含历史项） | 功率估计 W；`J_nom = Σ P̂·0.1 s` | 只凭命令预测"理想跟踪下应该耗多少" | **不是**物理 rollout 的结果 |
| **α（成本裕度）** | 校准集的配对样本 `ε_c = (J_exec − J_nom)/J_nom` | 一个标量（0.9 分位） | 把成本放大成 `(1+α)c`，让预算留够超支 | 不是"平均超支"，是分位；不保证单条不超 |
| **δ（几何裕度）** | 校准集的配对样本 `ε_d = max_t ‖p_exec − p_nom‖` | 一个标量（0.9 分位） | 把危险集一次性膨胀 `F ⊕ disc(δ)` | 不是安全距离，也不是"不会撞"的保证 |
| **独立检验** | 冻结后的 α/δ + 从未参与校准的 trial | 覆盖率 + 95% 区间 + 分组 | 报告"这套裕度在新条件下还管不管用" | 不是调参依据；覆盖率 ≠ 导航成功率 |

### 5.3 三个必须分开的对象（最容易混的一处）

| 对象 | 怎么得到 | 用途 | 不能拿来做什么 |
|---|---|---|---|
| **确定性名义模型** `f̄_cl` | 用同一套**实际施加**的速度命令做平面积分 | 生成名义轨迹 `p_nom`、算 `J_nom` | 不能用 clean 物理 rollout 顶替 |
| **clean 物理执行** | 关掉摩擦/质量/推力/观测噪声后真跑仿真 | 作为配对中的执行侧、功率模型拟合数据 | 它不是名义模型：跟踪误差仍在（v2 实测 `ε_d` 中位 1.1 m） |
| **扰动物理执行** | 五种 profile 打开扰动后真跑 | 校准 α/δ 的主样本 | 不能与 clean 混在一套统计里 |

### 5.4 为什么换控制器必须重做全部校准

α、δ、功率模型、足迹、步成本**都不是系统的固有常数**，而是"某个控制器 + 某种执行配置"下的经验分布：

- 换一个 actor → 关节动作分布变 → 力矩与关节速度变 → 功率模型系数变；
- 换 PD 增益/缩放/时序 → 跟踪误差分布变 → `ε_c`、`ε_d` 变 → α、δ 变；
- 换资产或碰撞体 → 足迹变 → 危险集膨胀的几何基准变。

所以 v2 的任何数值都不能标注成 HiRA 的结果。这也是本轮要"独立重复一整套"而不是"复用 v2 的裕度"的原因。

### 5.5 论文旧口径 vs 当前实现（只记差异，稿件不动）

| 项 | 论文（1029 稿） | 当前实现（v2 校准 + 本轮 HiRA 计划） |
|---|---|---|
| 成本 `J` 的单位 | 无量纲导航代理成本，**a.u.** | **正关节机械功，焦耳（J）** |
| 成本算法 | 代理成本（与代价地图/距离相关） | `Σ_k Σ_j max(τ_j·q̇_j, 0)·0.005`，逐关节截断后求和 |
| 预算量级 | 120 / 140 / 190 a.u.；部署 230 a.u. | `(1+α)·J_nom`，量级数百 J |
| 时间窗 | 60 s horizon | **固定 30 s** |
| 控制频率 | 50 Hz / 5 Hz（策略/上层） | **200 Hz 物理 / 50 Hz 策略 / 10 Hz 外部命令** |
| 裕度来源 | 论文为示例/设定值 | 由 800 条校准数据经验分位估出，且冻结后可复算 |

**差异只记录，不据此修改稿件**；稿件同步始终只出清单交用户。

### 5.6 本轮范围声明（写死，防止越界解读）

- 本轮**只新增一个控制器的校准实验**（HiRA 底层 actor 在 Isaac Lab 中的接入 + 独立重复 α/δ 校准）；
- **不做**：导航策略训练、真机/sim2real、稿件修改、重训底层、替换 v2；
- **不得**把"覆盖率达到 90%"解释成"导航成功概率 90%"或"形式化随机安全保证"——覆盖率只描述"在本次校准条件代表的执行条件下，误差落在裕度内的经验频率"；
- 完成校准 ≠ 完成导航算法，更 ≠ 安全到达证明。

---

## 6. 现有实现的复用地图与必须隔离项（HIRA-003）

完整机器可读版见 `reports/existing_pipeline_map.json`（20 个阶段，逐函数与字段）。此处只写结论。

### 6.1 一条链路的实际文件/函数（不是 README 复述）

```
协议 Protocol.load → 时间轴 Timeline.from_protocol → 命令 build_program/TableProgram.at
→ 名义轨迹 nominal.integrate → profile sample_applied/configure_env_for_profile
→ 采集 collect_trials.py（逐 env 窗口） → 记录 sim/recorder.TrialRecorder
→ 能量 energy（D-ENERGY-001） → 配对配对 pairing.compute_trial_errors
→ 存储 store.TrialStore → 对账 check_collection.py
→ 功率 fit_power_model.py/power.py → 足迹 footprint.py/geometry.py
→ 步成本 step_cost.py → 裕度 margins.py/fit_margins.py
→ 检验 coverage.py/evaluate_heldout.py → 近障 obstacles.py/near_obstacle.py
→ 审计 audit_all_splits.py → 锁 lock.py/build_data_lock.py
→ 交付 build_delivery_bundle.py/verify_delivery_bundle.py
```

### 6.2 v2 的隐式绑定（HiRA 侧必须显式覆盖，否则静默混用）

| 位置 | 值 | 后果 |
|---|---|---|
| `config.py` `DEFAULTS['CAMRA_LOG_ROOT']` | `/pub/data/caohy/logs/camra` | 写进 v2 数据树 |
| `config.py` `CAMRA_DEVICE` / `collect_trials --device` | `cuda:0` | GPU0 有他人 katago，曾致 Isaac 启动卡死 |
| `config.py` `CAMRA_TMUX_SESSION` | `camra-locomotion` | 日志归错会话 |
| `config.py` `CONTROLLER_VERSION` | `camra_go2_velocity_it2999_v2` | HiRA 包路径被拼错 |
| `protocol.py` `DEFAULT_VERSION` | `1.0.2` | 沿用 v2 协议 |
| `lock.py` `enforce_collection` 缺省 margins | `config/margins_v1.json` | 用 v2 裕度放行 HiRA |
| `build_data_lock.py` `tree_hashes(cal, rglob('*.py'))` | v2 全部 54 个 py | 新增 HiRA 代码会改 v2 锁树散列 |
| `collect_trials --allow-overwrite` | 默认 False（正式 split 一律 REFUSED） | 这是保护，沿用不改 |

**结论**：工具链可按「同一形态、独立实例」复用；上述 8 项默认值必须逐个覆盖。

### 6.3 既有事故与对 HiRA 的启示

| 事故 | 处理 | 对 HiRA |
|---|---|---|
| 整批连坐终止（155/160 被截断） | 逐 env `stop_ctrl` + 各自切片 + float64 复算（D-010） | 采集器沿用 |
| early reset 污染 | 窗口结束才允许 reset | 适配层需复核 |
| 模型散列含 `frozen_at` | 改为只散列内容 | 沿用同一规范 |
| 低速/停滞边界 | v2 接受并标注（ε_d 中位 1.1 m） | **不能假设 HiRA 相同，须独立测** |
| 近障 138/197 重叠 | 报告为「δ 必要非充分」 | 须重测 |
| 最高 attempt 计数 + 一次性补采 | v2 特例，不推广 | **禁止**（R-06：失败留主分母） |

### 6.4 可直接复用 vs 必须隔离

- **复用**：命令程序、名义积分、profile 定义、记录器与能量公式、配对误差、统计口径、门禁机制
- **隔离**：控制器包、协议版本、功率模型、足迹、α/δ、数据锁、trial_id/run_id/输出目录，以及 6.2 的 8 项默认值

---

## 7. HiRA 来源策略的执行语义（HIRA-004）

机器可读版见 `reports/hira_source_trace.json`。以下只写关键结论与必须处理的三件事。

### 7.1 载入与推理

- 载入：`on_policy_runner.py:221 load()` → `actor_critic.load_state_dict(...)`；推理：`:229 get_inference_policy()` → `act_inference()`
- **确定性**：`act_inference` 直接返回 `actor(obs)` 的均值，不采样（`std` 只训练期用）
- **无观测归一化、无循环状态、无历史缓存**（全库 grep 不到 running_mean / EmpiricalNormalization / LSTM）
- 网络：45 → 512 → 256 → 128 → 12，ELU，最后一层无激活（已用 checkpoint 张量核对）

**一处必须记住的矛盾**：文件名 `model_18500.pt`，但 checkpoint 里 `iter = 16000`，且 `infos = None`。
所以**文件名不能当训练步数证据**，训练期覆盖过的配置也无法从 checkpoint 恢复。

### 7.2 观测（45 维，逐段已核）

| 切片 | 量 | 缩放 |
|---|---|---|
| 0:3 | 机身角速度 | ×0.25 |
| 3:6 | 机身系投影重力 | ×1 |
| 6:9 | 命令 `[vx, vy, wz]` | ×2 / ×2 / ×0.25 |
| 9:21 | `q − q_default` | ×1 |
| 21:33 | `q̇` | ×0.05 |
| 33:45 | **上一策略步裁剪后的动作** | ×1 |

不含机身线速度，无额外历史拼接。观测在**每个策略步末尾**计算一次；分层路径在每次低层调用前还会再算一次。

### 7.3 动作与力矩

`clip(a, ±100)` → `q_target = 0.25·a + q_default` → `τ = clip(20·(q_target − q) − 0.5·q̇, ±URDF_effort)`，
**每个物理子步重算一次**（`decimation=4`）。`torque_limits` 逐关节取自 URDF 的 `effort`。

### 7.4 命令管线——最危险的一段

```
外部写入 commands[:, :3]
   ↓  每个策略步（不是只在重采样时）
legged_robot.py:307-310  commands[:,2] = clip(0.5 * wrap_to_pi(commands[:,3] − heading), −1, 1)
   ↓  每 10 s 对到期 env
legged_robot.py:326      commands[:, :2] *= (‖commands[:, :2]‖ > 0.2)
   ↓
观测第 6:9 维（策略看到的是被改写后的值）
```

两条改写：

1. **wz 每步被 heading 误差覆盖**——因为基类 `heading_command=True` 而 `GO2RoughCfg` 没覆盖；
2. **平面速度模长 ≤ 0.2 m/s 直接清零**——形成低速死区。

**这两条必须在适配层处理**：`heading_command=False`；死区要么关掉、要么显式记录为受控条件。
否则「外部命令」这个说法不成立，后面所有命令相关的统计都会偏。

### 7.5 reset 语义

- 关节角按 `q_default × rand(0.5, 1.5)` 随机化；初始高度 1 m 落下；episode 20 s
- reset 时 `self.actions` 清零 → 观测第 33:45 维在 reset 后为 0
- 与采集要求"固定起点、同 seed 可复现"冲突，适配层必须替换

### 7.6 三类清单（HIRA-004 的验收要求）

**A. 代码可确认**：网络结构与激活、推理用均值、无归一化、`clip_actions=100`、`action_scale=0.25`、
`p=20 / d=0.5`、URDF 力矩限幅、每子步 PD、45 维六段与缩放、每步改写 wz、0.2 m/s 死区、reset 随机化。

**B. 需运行时确认**：`state_dict` 键名与适配器是否完全匹配；URDF `effort` 逐关节值 vs Lab 导入后的 motor effort；
四元数 `(w,x,y,z)` 约定在 Lab 侧投影重力是否一致；`default_dof_pos` 的源顺序 ↔ Lab 顺序置换；
`clip_observations=100` 在 Lab 侧是否启用；`high_level_action_repeat=5` 与 10 Hz 外部命令的相位关系。

**C. 缺失的历史配置**：训练期实际命令课程范围、domain randomization 参数、噪声配置、随机种子
（`infos=None` 无法恢复）；`model_18500` 与 `iter=16000` 不一致，训练步数无法单独确认；
训练期是否覆盖 `action_scale / stiffness / damping` 无法验证。

### 7.7 对 R-01..R-04 的更新

| 风险 | 状态 | 说明 |
|---|---|---|
| R-01 heading 改写 | **confirmed（比预期更糟）** | 不是只在重采样时，而是每个策略步都覆盖；另有 0.2 m/s 死区 |
| R-02 每子步 PD | confirmed | 代码路径已确认 |
| R-03 reset 随机化 | confirmed | 0.5–1.5 倍 + 1 m 落下 |
| R-04 action_scale 双源 | confirmed | 解析值 0.25 |

"confirmed" 指**源侧行为已确认**；我们的适配层是否做对，仍由 HIRA-014/016 的数值对照判定。

---

## 8. 环境就绪（HIRA-006）

机器可读版见 `reports/environment_preflight.json`。

### 8.1 结论

| 项 | 结果 |
|---|---|
| 解释器 | `/pub/data/caohy/miniconda/envs/env_isaaclab/bin/python`（Python 3.11.15） |
| 核心包 | torch 2.7.0+cu126 / isaacsim 5.1.0.0 / isaaclab 0.47.2 / numpy 1.26.0 / gymnasium 1.2.0 / rsl-rl-lib 3.0.1 |
| CUDA | `is_available=True`，进程内可见 1 块 A100（物理 GPU2） |
| **最小启动** | **通过**：`ISAACLAB_MIN_OK`，app 启动 60.7 s，20 步 0.02 s，dt 0.005，device cuda:0，退出码 0 |
| 环境改动 | **零**：未安装/升级/卸载任何包；未装 Isaac Gym（符合计划） |
| 存储 | `/pub` ext4，29 T 用 27 T，剩 1.3 T |

日志：`/pub/data/caohy/logs/camra_hira/env_setup/env_min_start.log`，跑在 tmux `camra-hira-calibration:env`。

### 8.2 GPU 映射（容易搞错的一点）

`CUDA_VISIBLE_DEVICES=2` → 进程内只有 1 块可见卡 → 代码里写 `cuda:0` 指的就是**物理 GPU2**。
物理 GPU0 是他人两个 katago，GPU1 是他人 python；v2 期实测 GPU0 会让 Isaac 启动卡死。

### 8.3 路径

| 变量 | 值 | 权限 |
|---|---|---|
| `CAMRA_HIRA_DATA_ROOT` | `/pub/data/caohy/logs/camra_hira` | 可写（新建） |
| `CAMRA_HIRA_BUNDLE_ROOT` | `/pub/data/caohy/camra_hira_bundles` | 可写（新建） |
| `CAMRA_HIRA_EXPERIMENT_ROOT` | `.../experiments/hira_go2_18500` | 可写 |
| `CAMRA_V2_READONLY_LOG_ROOT` | `/pub/data/caohy/logs/camra` | **只读策略**（v2 的 68 GB） |
| `CAMRA_HIRA_SOURCE_ROOT` | `Paper/HiRA` @ 6a05af55 | 只读子模块 |

配置样例：`config/hira.env`（换机器只改这一份，源码不硬编码绝对路径）。

### 8.4 两条运行期硬要求（v2 期踩过的坑）

1. `PYTHONDONTWRITEBYTECODE=1` —— 否则 import 写 `.pyc` 落到 NFS 会卡死；
2. `HOME=/pub/data/caohy/fakehome` —— 否则 Isaac 往 `~/.nvidia-omniverse/logs` 写日志也会卡死。

`env -i` 启动时还要带 `TMPDIR / XDG_*` 指向本地盘。

### 8.5 告警判断

`Skipping NVIDIA GPU due CUDA being in bad state` 出现 4 次。**判为良性**：v2 期成功跑完 800 条的日志里同样有；
本次物理 0.02 s 跑完 20 步、device 报 `cuda:0`、退出码 0。每次运行仍记录该告警，若出现 util=0 且无进度，
再按脏 CUDA 上下文处理（换卡或清残留进程）。

### 8.6 归因规则（写死）

依赖缺失、CUDA 异常、NFS 卡顿一律记为**环境问题**并记录，**不得写成控制器性能失败**。

---

## 9. 来源与旧证据核对（HIRA-007）

机器可读版：`reports/source_artifact_manifest.json`、`reports/legacy_evidence_availability.json`。

### 9.1 来源身份

| 对象 | 值 | 与计划 |
|---|---|---|
| HiRA 子模块 | `6a05af55`，tree `fded2ec9`，dirty 0 | 一致 |
| actor checkpoint | `d30e5212…`（4,613,734 B），45-512-256-128-12 | **一致** |
| export `policy_1.pt` ×2 | `6d6f4d36…` / `0069ce66…` | 一致，**但只作身份核对，不作主路线** |
| URDF | `5fea7353…`（25,386 B），引用 7 个 `dae/*.dae` | 7 个 mesh **全部在位**（25 MB） |

三个 PPO 文件字节互不相同。按计划，主路线只从指定 checkpoint 重新提取 actor；两个导出不互换、不替代。

### 9.2 旧证据：哪些能复算，哪些只有报告

**可独立复算（本机有原始数据）**：逐 trial 能量（audit 已用独立 numpy 复算 1820 条，最差 1.26e-9）、
α/δ（从 800 条重算）、名义功率模型（从 400 条重拟合）、足迹（从 clean 帧重建）、
heldout 与近障覆盖率、近障几何重叠。

**仅有报告、无法复算**：TASK-036 底层复测结论（`overall_pass=False`）、TASK-024/026 训练期评估数字、
论文示例数值、候选包 `r4_it2999_v1` 的 JIT/ONNX 内容。原因：原始日志（662 MB / 10.6 MB / 1.3 GB）
在旧设备 `.12`，本机没有。

交付包复验：`ok=true`、20 项散列全对、α=0.19048279965649906、δ=2.6138664115280967。

### 9.3 引用 v2 时必须带上的两个注脚

1. `baseline_validate` 目录 111 个 = 80 个 trial + 31 个 `__a2`（D-007 批级截断重采）；
2. `near_obstacle` 目录 202 个 = 200 个 trial + 2 个 `__a2`（D-013 一次性补采）。

按最大 attempt 去重后才是 80 与 200。**这两个重试特例在 HiRA 新契约下不可复制（R-06）**，
将来引用 v2 数字作对照时，必须同时说明其分母含历史重试。

### 9.4 资产提醒

URDF（源，Isaac Gym 用）与 USD（Isaac Lab 用，`/pub/data/caohy/unitree_model/Go2/usd`）
**不是同一份资产**。质量、惯量、碰撞体可能不同，HIRA-012 必须逐项对照并记录差异。

---

## 10. 接口契约已冻结（HIRA-008）

- 契约：`config/hira_source_contract_v1.json`（内容散列 **`41f795630c23eff8…`**，侧车 `.sha256`）
- 差异报告：`reports/source_target_contract_diff.json`（16 项，match 6 / 必须覆盖 4 / unknown 5 / 设计上不等 3）

### 10.1 12 个驱动关节（源 URDF，已与 USD 交叉核对）

| 组 | 轴向 | 限位 (rad) | effort (N·m) | 默认角 |
|---|---|---|---|---|
| hip（四个） | [1,0,0] | ±1.0472 | 23.7 | FL/RL +0.1，FR/RR −0.1 |
| thigh FL/FR | [0,1,0] | −1.5708 … 3.4907 | 23.7 | 0.8 |
| thigh RL/RR | [0,1,0] | **−0.5236 … 4.5379** | 23.7 | 1.0 |
| calf（四个） | [0,1,0] | −2.7227 … −0.83776 | 35.55 | −1.5 |

**前后腿 thigh 限位不同**，URDF 与 USD 两侧一致（≤1e-3 rad）。任何"四条腿对称"的假设都是错的。
URDF 总质量 15.017 kg（base 6.921 + 4×2.024）。

### 10.2 三条必须在目标侧覆盖（源的行为不能照搬）

| 项 | 源行为 | 目标要求 | 原因 |
|---|---|---|---|
| `heading_command` | True，每策略步改写 `commands[:,2]` | **False** | 否则外部 yaw 命令到不了策略 |
| 0.2 m/s 死区 | 平面速度模长 ≤0.2 清零 | 关闭，或显式记为受控条件 | CaMRA 需要 0.1–0.3 低速层 |
| 初始姿态随机化 | `q_default × rand(0.5,1.5)`，1 m 落下 | 固定姿态与高度 | 采集要求同 seed 可复现 |

### 10.3 五项 unknown（不填 v2 值，指向具体任务）

| # | 未知 | 解除任务 |
|---|---|---|
| U-1 | USD 侧逐关节 effort（静态几何未记录） | HIRA-012 |
| U-2 | Lab articulation 的 DOF 顺序 | HIRA-013 |
| U-3 | 四元数约定（xyzw vs wxyz） | HIRA-013 |
| U-4 | Lab 侧是否启用 `clip_observations=100` | HIRA-013 |
| U-5 | 训练期覆盖（课程/随机化/种子） | **不可解除**（`infos=None`），只能以源码值 + 权重身份为准 |

### 10.4 设计上不等、只能量化的三项

执行器模型（手动 set torque vs Lab actuator）、接触与求解器（Gym PhysX vs Sim PhysX 5）、
资产（URDF vs USD）。这三项**不声明等价**，在 HIRA-012/014/017 里逐项量化残差。

### 10.5 禁止事项（写进契约的 prohibitions）

不得把 v2 的 `deploy.yaml` 当 HiRA 契约；不得沿用 v2 的增益/默认姿态/观测缩放；
不得因维度相同就认为相容；不得引入未记录的归一化；unknown 不得用 v2 值静默填充。

---

## 11. 接入验收已预登记（HIRA-009）

- 预登记：`config/integration_acceptance_prereg_v1.json`（内容散列 **`459b596017a4b7f7…`**，侧车 `.sha256`）
- 预算：`reports/resource_budget.json`
- **状态：在任何接入测试运行之前冻结。看到结果后不得放宽阈值。**

### 11.1 容差（工程验收阈值，不是"已测结果"）

| 类别 | atol | rtol | 用途 |
|---|---|---|---|
| 纯公式 float64 | 1e-12 | 1e-12 | q_target、PD、能量累加 |
| 观测 float32 | 1e-6 | 1e-5 | 45 维逐块对照 |
| actor CPU | 1e-5 | 1e-4 | 导出 vs 权重 |
| actor GPU | 1e-4 | 1e-4 | 含 fp32 非确定性 |
| q_target | 1e-6 | 1e-5 | 关节目标角 |
| 力矩 | 1e-4 | 1e-5 | 含 effort 限幅 |

另有：关节限位一致性 1e-3 rad、逐链接质量 1%。

### 11.2 15 项测试矩阵

T-01 身份与载入｜T-02 资产/顺序/坐标系｜T-03 观测与归一化｜T-04 动作/PD/力矩｜T-05 三时钟｜
T-06 命令不被改写｜T-07 reset 与上一动作｜T-08 clean 重复与批量隔离｜T-09 五 profile 实际施加｜
T-10 200 Hz 计量非侵入｜T-11 v2 拒混与复算｜T-12 CPU 同状态 oracle｜T-13 映射破坏必被拒｜
T-14 clean 闭环行为｜T-15 回归破坏必被拒

每项都有：输入、重复数、比较基准、判据、证据路径（`reports/integration_gate.json#T-xx`）、失败动作。

### 11.3 闭环序列（11 个工况 × 3 重复 = 33 条，上限 40）

零速、正/反 vx=0.5、左/右 vy=0.3、正/反 wz=0.5、弧线、低速 0.1/0.2/0.3。
判定：摔倒（高度 <0.15 m 或倾角 >60°）、方向符号、中速跟踪误差 ≤0.25 m/s、力矩饱和率 ≤30%。

**低速单列**：跟踪差不得算接入 bug，也不得因此删除命令域。

### 11.4 失败动作分两类

- T-01..T-13、T-15 任一失败 → **G1 不通过，pilot 与采集不得开始**；
- T-14 失败 → 先归因：能定位到映射/缩放/时序的接入错误 → 阻断；策略能力不足 → 记录并继续（不删域、不改配额）。

### 11.5 预算（估算，pilot 后必须重估）

| 项 | 估算 |
|---|---|
| 诊断试验 | 35–60 min（33 条闭环 + 15 条短试验 + 8 次进程启动，每次启动 60.7 s） |
| 正式采集 1820 条 | 中位 **2.5 h**；低负载 1–1.5 h；高负载可达 5–7.5 h |
| 存储 | 约 70 GB（余量 1.3 TB，占用 <6%） |
| 费用 | 0（全本地） |

暂停触发：吞吐低于 1 条/10 s 持续 10 分钟；剩余磁盘 <500 GB；诊断超 90 分钟；需放宽任何已登记阈值。

---

## 12. 显式配置入口与拒混（HIRA-010）

### 12.1 新增位置

- `CaMRA/locomotion/adapters/hira_go2/experiment.py`：`HiraExperiment` 配置对象 + `ConfigError`
- `CaMRA/calibration/experiments/hira_go2_18500/tests/check_config_isolation.py`：10 项自测
- 报告：`reports/config_isolation_tests.json`（**10/10 通过**）

### 12.2 设计要点（为什么不是"加个 if"）

`HiraExperiment` 的 11 个资源字段**没有默认值**，缺一个就抛 `ConfigError`；`from_env` 只认
`HIRA_*` 前缀。这样"忘了给"和"给了 v2 的"是两种不同的失败，且都在启动仿真之前发生。

三类检查分开：

| 检查 | 抓什么 | 例 |
|---|---|---|
| `identity_problems` | controller_id / experiment_id 被改成 v2 | `camra_go2_velocity_it2999_v2` |
| `v2_contamination_problems` | 任一字段路径命中 v2 冻结物或 v2 日志根 | 协议、模型、足迹、margins、锁、地图 |
| `existence_problems` | 文件/目录缺失、目录不可写 | 无 fallback，缺就是缺 |

输出路径也只能落在 `data_root` / `bundle_root` 下，`run_id` 含 `../` 或 `/` 直接拒绝。

### 12.3 自测覆盖（10 项）

U1 缺环境变量必失败｜U2 controller_id 指向 v2｜U3 protocol 指向 v2｜U4 margins 指向 v2｜
U5 data_root 指向 v2 日志根｜U6 正常配置零问题｜U7 缺 hira 产物时报缺失而非 fallback｜
U8 run_id 路径穿越｜U9 v2 旧产物散列未变｜U10 指纹含真实 checkpoint 散列（`d30e5212…`）

**U7 最有意义**：此刻 `hira_*` 协议/模型/裕度/地图/锁都还没生成（HIRA-023/026/029 才产出），
`validate(require_files=True)` 如实报 7 类缺失——这正好证明它**没有偷偷回退到 v2 的那一套**。

### 12.4 自测抓到的一个真 bug

第一版把 `power_model_v1.json` 这类名字直接当标记，结果 `hira_power_model_v1.json`
被自己的规则误判成 v2（U6 直接失败）。修正为带 `/config/` 前缀的精确标记，
并把这条教训写进代码注释——**标记要盯路径语义，不能只盯文件名后缀**。

### 12.5 旧证据未受影响（acceptance 第 2 条）

```
repo  v2 锁 b214809b…  n_bad=0   代码树 ff6631de…  54 个文件（未变）
/pub/data/caohy/camra_run/calibration  锁 b214809b…  n_bad=0  ← 历史复算用的只读快照
```

新增 adapter 代码放在 `locomotion/adapters/`，**不在** v2 锁的 5 个代码目录内，
所以旧锁的代码树散列不变、旧判定可复算。这正是"新实验用自己的锁、不覆盖旧锁"的实现前提。

---

## 13. 确定性 actor 的导出与数值对照（HIRA-011）

### 13.1 产物

| 产物 | 位置 | 要点 |
|---|---|---|
| 导出器 | `CaMRA/locomotion/adapters/hira_go2/export_actor.py` | 按路径加载源 `actor_critic.py`，不 import `rsl_rl` 包 |
| 推理物 | `exported/policy.pt` | TorchScript `script` 模式，768 018 B，189 324 参数，只含 actor |
| 来源清单 | `exported/policy_manifest.json` | 检查点散列、结构、逐张量散列、归档散列、规则说明 |
| 对照报告 | `reports/actor_export_parity.json` | 29 个用例（9 输入库 + 20 随机组）CPU/GPU 全过 |

### 13.2 结构是反推出来的，不是抄来的

第一版最容易犯的错是照抄 `LeggedRobotCfg` 的默认 `actor_hidden_dims = [256, 256, 256]`。
实测这个检查点是 **45→512→256→128→12**（`go2_config` 里被改过）。所以导出器从
`state_dict` 的形状反推宽度，再与源配置文本解析出的 `actor_hidden_dims` 交叉核对；
两边不一致直接失败。激活函数同理：名字从源配置读、构造函数从源模块的
`get_activation` 取，不自己写一个 `nn.ELU()` 假装等价。

### 13.3 三个"不"

- **不用 `act()`**：`act()` 里是 `Normal(mean, mean*0 + std).sample()`，带随机数；
  导出走 `act_inference`，即 `self.actor(obs)`。
- **不用 `strict=False`**：键缺失、键多余、形状不符全部直接失败。载入后再逐键比一次形状，
  防"键同名但被 reshape"。
- **不改权重**：检查点以只读打开，导出前后各算一次 sha256，不一致立即失败（实测一致，
  `d30e5212…`）。

### 13.4 数值结论

| 对照 | 最大绝对差 | 结论 |
|---|---|---|
| CPU：导出 vs 源 actor（9 用例，batch 1/2/4/8/32/45） | **0.0** | 逐位一致 |
| CPU：独立二次构造 vs 源 actor | **0.0** | 不是共享对象引用带来的巧合 |
| CPU：20 组随机观测 | **0.0** | T-12 的 repeats=20 已覆盖 |
| GPU：导出 vs GPU 源 | **0.0** | 同一图，同 kernel |
| GPU 源 vs CPU 源 | 1.465e-3 | 见下 |

GPU 与 CPU 的差全部来自 fp32 非结合律：最大绝对差出现在 ±100 边界输入，
但该点输出量级约 **4 257**，相对差 **5.5e-7**，判定由预登记的 `rtol 1e-4` 主导。
这不是精度回退，也不是"把阈值调宽"——预登记里 `actor_gpu` 本来就是 atol 1e-4 + rtol 1e-4。

### 13.5 两个真发现

1. **文件名不等于训练步数**：`model_18500.pt` 内部 `iter` 是 **16000**。
   身份判定一律以 sha256 为准（`d30e5212…`），报告里把这一条如实记成
   `iter_consistency.consistent = false`，不假装对得上。
2. **TorchScript 归档字节不可复现**：同进程内再序列化一次，散列就变了
   （`721a891a…` vs `a9626591…`，`archive_bytes_stable_in_process = false`）。
   所以归档散列只能当"本次产物出处凭证"，**不能当内容标识**。真正的内容标识是
   `actor_tensor_digest = 99236fbc…`：它由名字排序后的 name+shape+dtype+bytes 拼成，
   重新载入导出物后逐张量比对一致（`reload_tensor_digest_matches_source = true`）。
   这条写进了 `hash_rule`，避免下游把归档散列当"权重指纹"用。

### 13.6 "未知 normalizer 问题为零"是怎么答的

不是"没看见就说没有"，而是逐键审计：状态字典只应出现 `actor.*` / `critic.*` / `std` 三类，
出现 `normalizer.*` / `running_mean` / `running_var` / `obs_rms` 任一子串即判定存在训练期统计量。
实测 `n_keys = 16`，三类之外 **0 个键**，`has_normalizer = false`、`degenerate_tensors = []`（无全零张量）。

于是观测缩放的位置也确定了：**不在策略里**，而是源 env `compute_observations` 组包时按
`obs_scales` 乘进去（契约 `observation.blocks` 的 0.25 / 1.0 / [2,2,0.25] / 1.0 / 0.05 / 1.0）。
这一句是 HIRA-012 之后写观测适配器时的前提。

---

## 14. HiRA 专属资产与执行器（HIRA-012）

### 14.1 产物

| 产物 | 位置 |
|---|---|
| URDF 解析器（纯标准库） | `CaMRA/locomotion/adapters/hira_go2/urdf_model.py` |
| 资产/执行器配置 | `CaMRA/locomotion/adapters/hira_go2/asset_cfg.py` |
| 静态审计与契约交叉核对 | `CaMRA/locomotion/adapters/hira_go2/asset_audit.py` |
| 静态门禁（无需 GPU） | `experiments/hira_go2_18500/tests/audit_asset_static.py` |
| 运行时审计（Isaac Lab） | `experiments/hira_go2_18500/tests/check_asset_runtime.py` |
| 卡点诊断 | `experiments/hira_go2_18500/tests/diag_asset_stage.py` |
| 报告 | `experiments/hira_go2_18500/reports/asset_actuator_audit.json`（R1–R8 全过） |

### 14.2 为什么不复用 v2 的 USD

v2 用的 `Go2/usd/go2.usd` 来自 unitree_ros 的 `go2_description`；HiRA 用的是
`legged_gym_go2/resources/robots/go2/urdf/go2.urdf`（29 个 link、16 个 fixed 关节，
带头部零件与脚部球）。两者外形都是 Go2，但**不是同一份资产**。
本轮直接用 `sim_utils.UrdfFileCfg` 从源 URDF 导入，USD 落在
`/pub/data/caohy/logs/camra_hira/assets/usd_cache`（5 个文件，base 18.8 MB）。

### 14.3 执行器：让 PD 只生效一次

源是 `default_dof_drive_mode = 3`（纯 effort），每个物理子步自己算
`τ = kp(q_des − q) − kd·q̇`，裁到 URDF effort 后写进仿真。Lab 侧对应：

```
IdealPDActuatorCfg(stiffness=0.0, damping=0.0,
                   effort_limit=<URDF effort>, effort_limit_sim=<同值>)
```

`stiffness=damping=0` 时它退化成 `applied = clip(τ_ff, ±effort)`，正好是"透传 + 裁一次"。
四条实测：

| 检查 | 结果 |
|---|---|
| USD 侧刚度/阻尼 | 全 0（`get_dof_stiffnesses/get_dof_dampings`） |
| 力矩透传 | 命令与施加的最大差 **0.0** |
| 裁剪 | 命令 ±1000 → 施加 **±23.7**（= URDF effort） |
| 隐式 PD 残留 | 关节偏离默认角 0.3 rad、命令 0 → 施加**恒 0** |

最后一条是关键：如果 `ImplicitActuatorCfg` 或 USD drive 还留着 20/0.5，
这里会立刻看到非零力矩。

### 14.4 重要发现：URDF 顺序 ≠ USD 顺序

```
URDF : FL_hip FL_thigh FL_calf | FR_hip FR_thigh FR_calf | RL_… | RR_…   （按腿）
USD  : FL_hip FR_hip RL_hip RR_hip | FL_thigh … | FL_calf …              （按关节类型）
```

置换（USD 第 i 个 = URDF 第 perm[i] 个）= `[0,3,6,9,1,4,7,10,2,5,8,11]`，双向唯一。
契约里写"order_target 不得假定等于 order_source"，这不是保守措辞——实测就是不一样。
HIRA-013 的观测/动作适配器必须按名字建双射，任何"按位置对齐"都会把 hip 的动作
打到 thigh 上。

### 14.5 质量与 link 合并

29 个 link → 19 个 body。被合并掉的 10 个是 `FL/FR/RL/RR_calflower{,1}` 与 `imu`、`radar`
（都是 fixed 子链接，质量为 0）；`FL/FR/RL/RR_foot` 和 `Head_upper/Head_lower` 保留为独立 body。
总质量 **15.019 kg**，与 URDF 逐链接求和相差 **1.1e-8**（纯 float32）。

### 14.6 契约里标的 `unknown_from_static_asset` 补上了

| 项 | 值 |
|---|---|
| effort | hip/thigh **23.7**，calf **35.55**（`get_dof_max_forces`） |
| velocity limit | hip/thigh **30.1**，calf **20.07**（`get_dof_max_velocities`） |
| 限位 | 与 URDF 差 ≤ 5.1e-7 rad（预登记阈值 1e-3） |

effort/velocity 的残差是 7.6e-7 / 1.5e-6，量级正好是 float32 表示误差，不是配置差异。

### 14.7 两个坑（都花了时间）

1. **`Articulation.reset()` 不写 `init_state`**。它只重置执行器和外力，初始关节角要由
   env 的重置事件显式 `write_joint_state_to_sim`。第一版在 `sim.reset()` 后直接读
   `joint_pos`，读到的是"0 角 + 重力下垂几毫秒"，误差 1.318 rad，看上去像
   `init_state` 没生效。真因是读数时机，不是配置。HIRA-013 的逐 env reset 必须补这一步。
2. **`simulation_app.close()` 在本机会卡住 >5 min**（headless、卡被多人共享）。
   第一次跑时误以为是 URDF 导入卡死，白等了一轮；后来用分阶段脚本
   （`diag_asset_stage.py`）证明导入只要 9 s，卡的是关停。现在报告落盘后直接
   `os._exit(code)`，并把这条登记为环境差异。

### 14.8 声明的引擎差异（不假定为零）

- 源 `num_velocity_iterations = 0`，PhysX 要求 ≥1 → 取 1
- 源 `collapse_fixed_joints` 会保留 `dont_collapse="true"` 的关节；Isaac Lab 的 URDF
  导入没有等价开关，实测合并结果如上（不产生 DOF、总质量不变，但 link 划分不同）
- 碰撞体里的 cylinder 在两边都被替换为胶囊（源 `replace_cylinder_with_capsule = True`）
- 源 thickness 0.01 → `CollisionPropertiesCfg(contact_offset=0.01, rest_offset=0.0)`
- isaacgym 与 PhysX 5.1 的接触/摩擦模型不同，**接触力数值不可假定相等**，
  这一点在 HIRA-014 的闭环行为里才会真正显形

---

## 15. 45 维观测适配器（HIRA-013）

### 15.1 产物

| 产物 | 位置 |
|---|---|
| 适配器 | `CaMRA/locomotion/adapters/hira_go2/observation.py`（只用 torch，不 import isaaclab） |
| 自测 | `experiments/hira_go2_18500/tests/check_observation_adapter.py`（O1–O13，13/13 通过） |
| 报告 | `experiments/hira_go2_18500/reports/observation_adapter_tests.json` |

### 15.2 三处最容易做错的地方，都用结构固定住

1. **关节顺序**：`JointOrder` 按名字建双射，存两份索引（目标→源、源→目标），
   构造时校验"重复/缺失/互逆"，不做任何按位置对齐。
2. **上一动作的语义**：存的是 `clip(a, ±100)`、**未乘 action_scale** 的动作。
   源里 `self.actions` 就是这个量，0.25 只在 `_compute_torques` 里乘。
   如果把 `0.25·a` 存进观测，模型看到的输入会和训练时差 4 倍。
3. **缩放只做一次**：命令进来已是 m/s、rad/s，适配器只乘 `[2, 2, 0.25]`；
   观测裁剪在最后做一次 `clip(±100)`。

### 15.3 顺序不是"顺便对齐"的问题

HIRA-012 实测的 Lab DOF 顺序是按关节类型分组的：

```
Lab : FL_hip FR_hip RL_hip RR_hip | FL_thigh … | FL_calf …
源  : FL_hip FL_thigh FL_calf | FR_hip …
```

自测里 O2 用一份**独立 oracle**（逐块标量实现，不复用适配器任何逻辑）对照，
第一次跑就抓到 20 个点越界、最大差 2.466——原因是置换方向写反了
（`target_to_source` 用了 `source_index_of_target`）。两个索引互为逆，
所以"往返一致"的测试照样通过，只有和 oracle 逐点比才暴露。
**这就是为什么 O3（可逆性）不能代替 O2（逐点对照）。**

### 15.4 13 项自测

| ID | 查什么 | 结果 |
|---|---|---|
| O1 | 六块 3/3/3/12/12/12 = 45 | 通过 |
| O2 | 与独立 oracle 逐点对照 | max_abs 1.5e-8，0 越界 |
| O3 | 双射可逆、重复/缺失被拒 | 通过 |
| O4 | 上一动作 = `clip(a,±100)` 未缩放 | 250 → 100，观测 33:45 逐位相同 |
| O5 | 改上一动作只影响 33:45 | 0:33 变化 0.0 |
| O6 | 观测裁剪 ±100，两侧与 oracle 一致 | 通过 |
| O7 | 逐 env reset 只清该 env | `[1,1,0,1]` |
| O8 | 噪声只落 9:33；clean 不加噪；无向量时加噪被拒 | 0:9 与 33:45 变化 0.0 |
| O9 | 只缩放一次（1.0→0.25、0.5→1.0、0.4→0.1） | 通过 |
| O10 | 恰 45 维、不含机身线速度 | 通过 |
| O11 | 投影重力 = R(q)ᵀ(0,0,−1)，单位长度 | 3 个姿态全对 |
| O12 | 世界角速度 → 机体系 | yaw90 下 (1,0,0)→(0,−1,0)，z 分量不变 |
| O13 | xyzw 当 wxyz 用会错 | 错序差 0.632，换序后 6e-8 |

### 15.5 四元数与角速度的约定

模块统一用 **wxyz**（Isaac Lab），函数名里写死 `_wxyz`；源 isaacgym 是 **xyzw**。
O13 的意义就是把这件事量化：同一个四元数按错顺序读，投影重力会差 0.632（远大于任何容差）。

`body_angular_velocity_wxyz` 也单独提供：源的 `base_ang_vel` 是机体系量，
Lab 侧拿到世界系角速度，必须 `R(q)ᵀ` 转一次。**世界 yaw_rate ≠ 机体系 z 分量**，
这两者在机器人倾斜时差别不小。

### 15.6 还没做的部分

观测适配器目前只覆盖"拼接与语义"。它与真实 Isaac Lab 状态张量的对接
（`robot.data.root_quat_w`、`root_ang_vel_w`、`joint_pos` 的取值时机）要在
HIRA-014 的闭环里做，那时才能验证"取的是物理子步后的值还是策略步前的值"。
本轮的 oracle 对照证明的是**公式与顺序正确**，不是"接进仿真后自动正确"。

---

## 16. 动作映射与每子步 PD（HIRA-014）

### 16.1 产物

| 产物 | 位置 |
|---|---|
| 动作/PD 适配器 | `CaMRA/locomotion/adapters/hira_go2/action_pd.py` |
| 自测 | `experiments/hira_go2_18500/tests/check_action_pd.py`（A1–A10，10/10） |
| 报告 | `experiments/hira_go2_18500/reports/action_pd_tests.json` |

### 16.2 一条控制链，三个"只发生一次"

```
策略步（0.02 s）：a_clipped = clip(a, ±100)        ← 只裁一次
                 q_target = q_default + 0.25·a     ← 只在入口算一次
物理子步（0.005 s × 4）：
                 τ_PD = 20·(q_target − q) − 0.5·q̇  ← 每子步用新鲜 q/q̇ 重算
                 τ = clip(τ_PD, ±effort)           ← 只裁一次
                 write(τ)
```

对应的三条强制：

- `set_policy_action` 收到超过 ±100 的动作**直接拒绝**——不在这里再裁，否则
  "观测里的上一动作"和"真正执行的动作"会在边界处分叉。
- `compute_substep` 必须每步传新的 `q/q̇`；`q_target` 在 4 步内恒定（A5 实测）。
- 力矩裁剪只此一处，A7 用 `clipped == clamp(PD)` 直接验证。

### 16.3 10 项自测

| ID | 查什么 | 结果 |
|---|---|---|
| A1 | `q_target = q_default + 0.25a` 逐关节 | 差 0.0 |
| A2 | 零动作 → 默认角 | 差 0.0 |
| A3 | ±100 动作 → τ 顶到 ±effort | τ_PD=500 → 裁到 23.7/23.7/35.55，12 关节全饱和 |
| A4 | 单关节 ±0.1 rad 扰动 | τ=∓2.000，其余关节 0.0 |
| A5 | 4 子步共用 q_target、τ 每步变 | 通过 |
| A6 | τ 按名字写回 Lab 位置 | 逐关节一致，置换往返差 0.0 |
| A7 | 超界动作被拒、力矩只裁一次 | 通过 |
| A8 | kp=20、kd=0.5（不是 v2 的 25） | 通过 |
| A9 | 六项日志齐全 | 缺字段 0 |
| A10 | 无隐藏速度限幅 | q̇=40 rad/s（>30.1 上限）时 τ=+20.0 未被裁 |

A10 值得单独说：如果哪里偷偷按 `velocity_limit` 限了力矩，40 rad/s 这个点会立刻
露出不同的数；实测它老老实实等于 `0.5×40`。

### 16.4 与 HIRA-013 的衔接

`ObservationAdapter.apply_action` 返回的裁剪后动作，直接喂给
`ActionTorqueAdapter.set_policy_action`，两边对"上一动作"的定义因此天然一致：
同一个 ±100 裁剪值，既进 33:45 观测块，也进 q_target。

### 16.5 仍然没做的

这一层只保证"给定 q/q̇ 与动作，算出的 τ 与源公式一致"。它**不保证** Isaac Lab 在
`write_joint_effort_target → write_data_to_sim → sim.step()` 之间取到的 q/q̇ 就是
子步真正使用的值——那是 HIRA-015 的同状态 oracle 对照与 HIRA-016 之后的闭环要回答的。

---

## 17. 独立来源 oracle 与同状态对照（HIRA-015）

### 17.1 产物

| 产物 | 位置 |
|---|---|
| 固定样本（9 条） | `experiments/hira_go2_18500/tests/reference_fixtures/fixtures_v1.json`（sha256 `eefc3718…`） |
| 样本清单 | `tests/reference_fixtures/MANIFEST.json`（记录文件散列与条数） |
| 样本生成器 | `tests/make_reference_fixtures.py`（手写数值，无随机数） |
| 独立 oracle | `tests/source_oracle.py`（**不 import `hira_go2`**） |
| 对照 runner | `tests/check_source_adapter_parity.py` |
| 报告 | `reports/source_adapter_parity.json` |

### 17.2 oracle 为什么算"独立"

| 项 | 被测适配器 | oracle |
|---|---|---|
| 四元数 | 叉乘展开 `v − 2w(q×v) + 2q×(q×v)` | 先建旋转矩阵，再做 `Rᵀv` |
| 关节顺序 | 索引置换数组 | 一律按名字查字典 |
| 数据 | torch（float32/float64） | Python float（双精度） |
| 裁剪 | `torch.clamp` | 逐元素 `min/max` |
| 输入依赖 | 读 URDF + 常量 | **只吃 fixture JSON** |

fixture 里自带默认角与 effort 上限，所以 oracle 连 URDF 都不读。两条实现唯一的共同输入就是这份 JSON。

### 17.3 九条样本

| ID | 场景 | 想卡住什么 |
|---|---|---|
| F01 | 静止 | 最基本的等价点（全零） |
| F02 | 倾斜（roll 12°、pitch −18°） | 重力投影落到两轴 |
| F03 | 转向（yaw 60° + z 角速度） | 世界 z 角速度 ≠ 机体系分量 |
| F04 | 关节速度 ±5 rad/s | 21:33 块非零且不裁 |
| F05 | 上一动作含 ±100 | 33:45 边界与零值混合 |
| F06 | 裁剪边界（命令 60/−60/300、位置偏 200 rad、速度 3000 rad/s、动作 ±100） | 观测与力矩同时饱和 |
| F07 | 关节置换（第三种打乱顺序表达同一状态） | 只能按名字对齐 |
| F08 | 侧倾 90° | 重力全落 y 轴，最容易暴露四元数转置/顺序错 |
| F09 | 低速巡航混合态 | 接近真实运行的量级 |

### 17.4 结果（设备 × 精度共四组）

| 配置 | 样本 | obs 最大绝对差 | q_target | τ | actor 输出 |
|---|---|---|---|---|---|
| cpu / float64 | 9/9 | 1.26e-8 | 0.0 | 0.0 | 2.4e-7 |
| cpu / float32 | 9/9 | 1.79e-7 | 7.6e-7 | 1.9e-6 | 2.4e-7 |
| cuda:0 / float64 | 9/9 | 1.26e-8 | 0.0 | 0.0 | 2.4e-7 |
| cuda:0 / float32 | 9/9 | 1.79e-7 | 7.6e-7 | 1.9e-6 | 2.4e-7 |

float32 一组才是"生产精度"：0.15 m/s 量级的观测、几十 N·m 的力矩，误差都在 1e-6 以内，
远低于预登记的 1e-6/1e-5 与 1e-4/1e-5。actor 输出差 2.4e-7 来自观测端那 1.8e-7 的放大，
量级仍然是一个 float32 ulp 级别。

**必须说清的一点**：F06 这一条的差是**恰好 0.0**，因为它所有分量都被裁到 ±100，
裁剪把细微差别抹平了。所以"饱和样本通过"不能单独当作精度证据——
真正的精度证据来自 F02/F04/F09 这些不饱和的样本，而"裁剪行为一致"由 F06 单独证明。

### 17.5 负测试（五类，全部被检出）

| ID | 故意做错什么 | 检出 | 暴露位置的最大差 |
|---|---|---|---|
| N1 | 按位置对齐（源顺序数组当目标顺序用） | 是 | 2.500 rad |
| N2 | 命令缩放 0.25 写成 0.5 | 是 | 0.125 |
| N3 | 四元数 xyzw 当 wxyz | 是 | 0.866 |
| N4 | RL_calf 默认角偏 0.05 rad | 是 | 0.050 |
| N5 | kp 用 v2 的 25 而非 20 | 是 | 3.750 N·m |

五个都检出的意义在于：这套对照**不是因为容差宽才通过**——任何一类真实错误都会立刻越界，
而且量级都在 1e-2 以上，比 1e-6 的容差高四个数量级。

### 17.6 过程中踩到的一个真 bug（记录在案）

同一个 TorchScript 模块连续 `.to("cpu")` 再 `.to("cuda:0")` 后，出现
`weight 在 cuda、input 在 cpu` 的诡异报错——参数设备元数据与实际权重不一致。
改成**每个设备单独 `torch.jit.load`** 后消失。这类问题只会在真正跑 GPU 时出现，
CPU 单测永远碰不到，所以"CPU 通过"不能替代"GPU 通过"。

### 17.7 边界

本轮证明的是**同状态数值等价**：给定同一份物理状态，两条实现算出同样的观测、目标角、力矩。
它**不**证明跨引擎长轨迹一致（isaacgym 与 PhysX 的接触/积分不同，本就要求不逐帧相同），
也不证明"仿真实测的 q/q̇ 与写入时序正确"——那要等 HIRA-016 起的闭环与时间轴检查。

---

## 18. 三时钟、外部命令与逐 env 终止（HIRA-016）

### 18.1 产物

| 产物 | 位置 |
|---|---|
| 最小闭环 runner | `CaMRA/locomotion/adapters/hira_go2/env_runner.py` |
| 验证脚本（三阶段） | `tests/check_timing_command_reset.py`（`--phase main/termination/merge`） |
| 报告 | `reports/timing_command_reset.json`（C1–C8 + N1–N3，11/11） |
| 逐步 trace | `reports/timing_command_trace.jsonl`（1500 行，每步命令/高度/倾角/力矩峰值） |

### 18.2 为什么分两次跑

一个进程里只能有一份场景：`InteractiveScene` 的 prim 路径固定是 `/World/envs/env_*`，
第二个场景会直接 `A prim already exists`。所以：

- `--phase main`：1 个 env，warmup + 30 s 窗口
- `--phase termination`：4 个 env，终止与跨 env 隔离
- `--phase merge`：纯 python 合并（不需要 torch / Isaac）

这也顺带证明了"批量与单 env 必须分进程验证"这条纪律——不是我们想偷懒。

### 18.3 时钟是数出来的，不是读配置

| 项 | 期望 | 实测 |
|---|---|---|
| warmup 物理步 | 200 | 200（策略步 0） |
| 窗口物理步 | 6000 | 6000 |
| 策略步 | 1500 | 1500 |
| 命令区间 | 300 | 300 |
| 配对点 | 301 | 301 |
| 相邻策略步步距 | 4 | `{4}`（唯一值） |
| 每区间推理次数 | 5 | `{5}`（唯一值） |

### 18.4 命令保真

窗口里循环施加 `[0,0,0] → [0.3,0,0] → [0.1,0,0] → [0,0.3,0] → [0,0,0.5]`：

- 观测命令块与 `[2vx, 2vy, 0.25wz]` 的最大偏差 **0.0**；
- `0.1 m/s` 那一档 `obs[6] = 0.200`，**没有被 0.2 死区清零**；
- 纯 `vy` 与纯 `wz` 命令都原样进入观测，**没有被 heading 反算或重采样吞掉**；
- 中途把命令从 0.5 改成 0.2，下一个策略步观测立刻从 1.0 变 0.4，**没有内部锁定**。

### 18.5 逐 env 终止

人为把 env0 压到 0.08 m：

- 终止记录里 `root_height = 0.098 m`，**是在 reset 之前抓的**（不是 reset 后的 0.42）；
- 只重置 env0：它的上一动作从 3.898 清零、root 回到 0.420 m；
- env1..3 的上一动作与命令**逐位不变**（差 0.0），观测命令块变化 0.0。

### 18.6 一个真 bug：`sim.step()` 不刷新 data 缓冲

第一版跑完 30 s 窗口后，`height_row0` 在 1500 步里**恒等于 0.420、倾角恒 0.0**，
看上去像"机器人完美静止"。实际是 `robot.data` 从来没更新——
`sim.step()` 只推进物理，`Articulation.data` 要靠 `robot.update(dt)` 刷新。
不刷新时，PD 用的是上一步之前的 q/qdot，观测量冻在初始值，
**整套控制是开环的**。

补上 `robot.update(self.timing.sim_dt)`（每个物理子步一次）后重跑：
高度 0.248–0.337 m 起伏、倾角最大 7.8°、力矩峰值 32.28 N·m（hip 的 23.7 已饱和）、
窗口内 0 次终止。时钟检查在两版里都通过——**这正是"只测计数测不出开环"的例子**，
所以报告里同时留了高度/倾角序列。

### 18.7 负测试

| ID | 故意做错 | 检出 | 说明 |
|---|---|---|---|
| N1 | decimation 写成 3 | 是 | 用合成轨迹验证检查器：步距集合 `{3}` ≠ `{4}`。真实重建第二个场景受 prim 路径冲突限制，已如实注明 |
| N2 | 命令写入后不生效（旧命令覆盖） | 是 | 观测在下一个策略步就跟着变，说明不存在内部缓存 |
| N3 | 重置 env0 污染其他 env | 是 | 真重置（env 数 1）后 env1..3 的命令块与上一动作逐位不变 |

### 18.8 边界

本轮窗口内的滚动/跟踪表现**不作结论**：命令序列是为时钟与命令通路设计的，
不是为运动能力设计的（HIRA-017 才按预登记的 11 条命令逐条评）。
这里唯一关于行为的事实是"30 s 内没有触发终止"，且 0.248 m 的低位值得在 017 里单独看。

---

## 19. clean 闭环评估与 C-08 归因（HIRA-017，**未通过**）

### 19.1 执行方式

11 条预登记命令 × 3 个固定 seed（11/22/33）× 30 s = 33 条，分两个分片并行
（`clean_closed_loop.shard{0,1}.json` → `clean_closed_loop.json`）。
每条 30 s 实际约 200 s，合计约 53 min 墙钟时间。

### 19.2 先修了一个**协议错误**（重要）

第一版在每条 trial 前跑了 200 个物理步的 warmup（零动作保持默认姿态）。
结果 `[0.5,0,0]` 只测到 0.315 m/s（误差 0.187），机器人高度全程 0.248 m。

诊断发现：零动作保持 1 s 会让机器人**先塌下去**（0.42 → 0.248 m），
策略接管时已经在训练分布之外，评估的其实是"从塌陷姿态恢复"。
改成 **warmup = 0**（源的行为就是 reset 后立刻由策略接管）后，
同一条命令测到 **0.469 m/s（误差 0.032）**。

这条差异不是小事：它把误差从 0.19 拉到 0.03，足以让结论从"不合格"翻成"合格"。
已把 `--warmup-steps` 默认设为 0，并在 HIRA-016 的报告里保留了 warmup 的用途（只用于时钟测试）。

### 19.3 结果（33/33 确定性一致）

| 命令 | 场景 | 实测 (vx, vy, wz) | 速度误差 | 最低高度 | 判定 |
|---|---|---|---|---|---|
| C-01 | 零速 | (+0.135, −0.020, −0.079) | 0.137 | 0.259 | ok（非门槛） |
| C-02 | 正向 0.5 | (+0.469, +0.009, −0.067) | 0.032 | 0.237 | ok |
| C-03 | 反向 0.5 | (−0.341, −0.003, −0.009) | 0.159 | 0.287 | ok |
| C-04 | 左移 0.3 | (+0.186, +0.158, +0.001) | 0.234 | 0.293 | ok（贴线） |
| C-05 | 右移 0.3 | (+0.130, −0.163, −0.003) | 0.189 | 0.292 | ok |
| C-06 | 正转 0.5 | (+0.079, −0.018, +0.494) | 0.081 | 0.289 | ok |
| C-07 | 反转 0.5 | (+0.102, +0.023, −0.443) | 0.104 | 0.314 | ok |
| **C-08** | **弧线 0.4 + 0.3** | **(+0.660, −0.122, +0.243)** | **0.288** | 0.242 | **FAIL** |
| C-09 | 低速 0.1 | (+0.152, −0.009, −0.049) | 0.053 | 0.254 | ok（能力边界，单列） |
| C-10 | 低速 0.2 | (+0.278, −0.023, −0.074) | 0.081 | 0.252 | ok |
| C-11 | 低速 0.3 | (+0.332, +0.000, −0.085) | 0.032 | 0.252 | ok |

- **无摔倒、无 NaN、无非有限值**；力矩饱和率全部 **0.0%**（阈值 30%）。
- 方向：C-02..C-07 全部符号正确（正反向、左右移、正反转）。
- 三个 seed 逐位一致（`determinism` 11/11 identical）：clean 确实没有隐藏随机化。
  代价是 33 条里每个命令只有 **1 个独立样本**，不能当 33 个独立点做统计。
- 附带事实：零速命令下仍有 **+0.135 m/s 前向漂移**（C-01 不是门槛，但记录在案）。

### 19.4 C-08 的归因（这一步是为了不把能力问题当接入错误）

`[0.4, 0, 0.3]` 实测 vx = 0.660，超出命令 0.26。做了两组对照：

**对照一：关节顺序假设**（怀疑"源顺序=URDF 声明顺序"错）
把顺序改成"源 = 引擎 BFS 顺序（即与 Lab 相同）"重跑诊断：
高度塌到 0.108 m、跟踪误差 22.9 rad、力矩顶到 35.55 上限。
→ **顺序假设正确，此路排除**。

**对照二：前向速度是否随 yaw 指令跳变**（同一 vx=0.4，只改 wz）

| 命令 | 实测 vx | 实测 wz |
|---|---|---|
| [0.4, 0, 0.00] | +0.364 | −0.079 |
| [0.4, 0, 0.15] | **+0.625** | +0.055 |
| [0.4, 0, 0.30] | +0.636 | +0.243 |
| [0.4, 0, 0.45] | +0.657 | +0.414 |
| [0.2, 0, 0.30] | **+0.013** | +0.022 |

同一前向指令，只要 **wz 从 0 变成 0.15**，前向速度从 0.364 跳到 0.625；
而 `[0.2, 0, 0.3]` 又几乎原地不动（vx=0.013）。
这说明策略在"带转向"的组合命令下会**切换步态**，不是线性缩放；
若存在缩放/映射错误，vx 会随命令整体偏移，而不是"wz 一出现就跳"。

**结论**：C-08 的失败**可归因到策略行为（能力/后端差异）**，不是本轮的映射、缩放或时序错误——
后三者已被 HIRA-015（同状态数值等价，误差 ≤2e-6）、HIRA-016（三时钟精确）、
以及上面的顺序对照逐条排除。

### 19.5 状态与阻塞

- 按 HIRA-017 的验收（"必需闭环测试满足预登记工程准入"），本任务 **status 保持 false**：
  C-08 的 0.288 > 0.25，预登记门槛未满足。
- 按预登记 `blocking_rules` 的 T-14 条款，能力不足应"记录并继续"，
  但这需要把 C-08 明确判为"能力"而不是"接入错误"，属于**判定口径**问题，
  按 `uncertainty_policy`（预登记门槛失败/需要改变口径时必须暂停询问）**已停下等用户裁定**。
- 后续 HIRA-018 依赖 HIRA-017，当前不再推进。

### 19.6 未受影响的结论

HIRA-011..016 的结论不依赖本轮闭环结果：导出、资产、观测、PD、同状态 parity、
三时钟与逐 env 重置都是独立验证过的。C-08 只影响"闭环行为"这一域。

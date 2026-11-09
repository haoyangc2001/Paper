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

---

## 20. 摩擦发现、017 重测与 018 结果

### 20.1 关键发现：导入后的摩擦不是源的值

HIRA-018 的第一次运行时读到 **导入后的材质摩擦是 0.5**，而源 URDF/isaacgym 的默认是 **1.0**。
HIRA-017 的 33 条 closed-loop 全部是在 **0.5** 下跑的，这是一个协议错误——
"clean" 在预登记里的定义就是无附加扰动、与源一致的物理条件。

用同一份代码在 friction=1.0 下重测两条命令，差异非常大：

| 命令 | friction 0.5 | friction 1.0 |
|---|---|---|
| C-02 [0.5,0,0] | 0.469（误差 0.032） | 0.445（误差 0.057） |
| C-08 [0.4,0,0.3] | **0.660（误差 0.288，曾判失败）** | **0.326（误差 0.087，通过）** |

也就是说，之前被归因为"策略步态切换"的 C-08 超速，**主因是脚底打滑**（摩擦只有源的一半）。

### 20.2 017 重测结果（friction=1.0，33 条）

| 命令 | 实测 (vx, vy, wz) | 速度误差 | 最低高度 |
|---|---|---|---|
| C-01 零速 | (−0.003, +0.035, −0.018) | 0.035 | 0.288 |
| C-02 正向 0.5 | (+0.445, +0.016, −0.008) | 0.057 | 0.312 |
| C-03 反向 0.5 | (−0.460, +0.036, −0.021) | 0.054 | 0.305 |
| C-04 左移 0.3 | (+0.035, +0.269, −0.019) | 0.046 | 0.306 |
| C-05 右移 0.3 | (+0.003, −0.262, −0.012) | 0.038 | 0.310 |
| C-06 正转 0.5 | (+0.019, +0.009, +0.490) | 0.021 | 0.289 |
| C-07 反转 0.5 | (+0.006, +0.007, −0.448) | 0.009 | 0.315 |
| C-08 弧线 | (+0.326, +0.045, +0.269) | 0.087 | 0.307 |
| C-09 低速 0.1 | (+0.082, +0.025, −0.002) | 0.031 | 0.296 |
| C-10 低速 0.2 | (+0.158, +0.047, −0.013) | 0.063 | 0.298 |
| C-11 低速 0.3 | (+0.227, +0.053, −0.009) | 0.090 | 0.307 |

**11/11 通过**：无摔倒、无 NaN、饱和率 0%、方向全部正确、误差全部 ≤0.09（阈值 0.25）。
零速命令的漂移也从 0.135 m/s 降到 0.003 m/s——打滑同样解释了这个"静止却在走"的现象。

旧结果保留为 `clean_closed_loop_fric050.json`，两份都在仓库里，可对照。

### 20.3 归属：这次不是"能力边界"，是我的协议错误

用户此前裁定"把 C-08 记为能力边界并继续"。事后证明 C-08 在正确摩擦下本来就过，
所以那条裁定**没有被用来放宽任何阈值**：阈值没动、也没有删命令域，
只是把条件改回了预登记里"clean"的真实含义（与源一致的摩擦）。

教训写进这里：**"能否复现源行为"取决于资产保真度，而不只是映射与时钟**。
HIRA-012 的资产审计只核对了质量、限位、effort、能力矩，**漏了材质摩擦**；
这一项必须在 HIRA-020 冻结控制器包时补上，否则交付包里的 clean 仍然是半摩擦条件。

### 20.4 018 五 profile 结果（9/9）

| 检查 | 结果 |
|---|---|
| P0 五个条件与协议同名 | clean / friction_only / mass_only / push_only / combined 全部存在 |
| P1 摩擦写入 0.3（仅 env0） | 27 个 shape 读回全 0.3；env1 保持 1.00 |
| P2 质量 +3 kg（仅 env0） | base 6.921→9.921 kg，总质量 15.019→18.019；env1 Δ=0 |
| P3 推扰 | 实测 Δv=(0.500, 0.000)，Δyaw=0.0；机制是 `write_root_velocity_to_sim`（速度冲击，不是牛顿力） |
| P4 观测噪声 | 0:9 与 33:45 变化 0.0；9:21 ≤0.0095（上限 0.01）、21:33 ≤1.49（上限 1.5） |
| P5 clean 确定性 | 两次运行末段指纹 **逐位一致**（在任何 profile 写入之前测；写入后再测会有 0.16 的漂移，因为写回质量/材质不能完全恢复求解器缓存） |
| P6 扰动确实生效 | friction Δ=6.5e-2、mass Δ=1.16e-1、push Δ=2.5e-2（相对 clean） |
| P7 恢复基准 | 质量/惯量/材质读回残差 0.0 |
| P8 跨 env | 只改 env0，env1 的质量与摩擦完全不变 |

### 20.5 顺带修掉的两个实现问题

1. **`reset` 后必须刷新 data 缓冲**（`robot.update(dt)`）。不刷新时，下一次观测/PD 用的是
   上一轮 trial 结束时的状态；这正是 018 的 P5 一开始失败的原因（0.163 的差），
   与 HIRA-016 里 `sim.step()` 不刷新数据是同一类问题。
2. **同进程内 reset 无法逐位复现**：写入质量/材质后再"恢复"并不会清掉 PhysX 的接触/求解器缓存。
   所以确定性检查放在任何 profile 写入之前做；跨进程逐位一致已由 017 的 33 条 trial 证明。

### 20.6 待办

- HIRA-020 冻结控制器包时，**必须把材质摩擦写入 bundle 的资产配置**（源 1.0），
  并把"导入默认 0.5"作为已修复缺口登记。
- HIRA-017/018 的结论只在 friction=1.0 下成立；后续采集也必须显式设置摩擦，不能依赖导入默认值。

## §21 HIRA-019：200 Hz 真实力矩与能量计量（17/17）

### 21.1 结论

`reports/hira_energy_meter_validation.json`：17 项检查全部通过（`verdict.all_passed=true`，
`failed=[]`）。核心数字：

| 项 | 实测 | 判据 |
|---|---|---|
| 实际施加 vs 我们写入的限幅力矩 | max\|Δ\| = **0.0**（2000 子步） | ≤1e-4 |
| computed_torque vs applied_torque | max\|Δ\| = **0.0** | ≤1e-4 |
| 在线（torch f64）vs numpy 复算 vs EnergyMeter 重放 | 三次相同，rel = **0.0** | ≤1e-9（T-10） |
| 记录器开/关整条轨迹 | 散列相同，逐元素 max\|Δ\| = **0.0** | 逐位一致 |
| 强制饱和时请求 vs 施加 | 10069.9 J vs 543.3 J（**18.5×**） | 必须能区分 |

### 21.2 力矩来源（E01–E04、E13）

`applied_torque` 不是"我们想施加的力矩"，而是 Isaac Lab 里这条链的产物，本次在运行时把
源码行号与文件散列一并抓进报告（`provenance` 段）：

- `articulation.py` L1865：`_joint_effort_target_sim[:, indices] = control_action.joint_efforts`
- 同文件 L1869：`_data.applied_torque[:, indices] = actuator.applied_effort` —— **同一份值**
- 同文件 L216：`set_dof_actuation_forces(self._joint_effort_target_sim, ...)` —— 写进 PhysX

运行期对照给出三条硬结论：实际施加与写入值逐位相同（执行器没有再改一笔）；
`computed_torque` 与 `applied_torque` 也相同（`IdealPDActuator` 的 kp/kd 都是 0，没有隐藏 PD）；
限幅只发生在我们这一层，且实际施加从不越界。

**请求力矩不能当施加值用**：动作 +100 时 τ_PD 达 500 N·m，实际施加被裁到 35.55 N·m，
两种口径算出的能量差 18.5 倍。这条是本任务最先要防的错误。

### 21.3 相位（E06）

同一段 10 s 数据，按子步**起始**速度算得 J = 198.070 J，按子步**结束**速度算得 212.161 J，
相差 **6.64%**（单步内 \|Δq̇\| 最大 14.44 rad/s）。契约取后者，与 v2 记录器 `sim/recorder.py`
的顺序逐字一致：先取 `applied_torque`（属于刚执行完的子步）→ `robot.update(dt)` → 取 `joint_vel`。
相位选错不会报错，只会让每个 trial 的能量系统性偏低 6.6%，所以这一条必须显式量出来。

### 21.4 逐关节截断（E07、E11）

错误口径（先对关节求和再截断）在同一段数据上给出 56.492 J，契约口径 212.161 J，**差 73.4%**——
关节间的正负功大量互相抵消。E11 用已知力矩直接证明：+3 N·m 把 calf 转起来后改施 −3 N·m，
其中 52 个"仍正向运动"的子步单步能量**逐位为 0**（若用 \|τ·q̇\| 或先求和再截断，这些步会非零）。

### 21.5 边界（E09、E10、E12、E14、E15）

- 零力矩 200 子步：J = `0.0`（逐位），机器人从 0.076 m 原地瘫下——符号基线成立。
- 已知单关节 +3 N·m、100 子步：J = 5.130677794，闭式值 Σmax(3·q̇,0)dt = 5.130677794（rel 1.7e-16），
  其余 11 个关节贡献**恰好 0**。
- 外部速度冲击 1 m/s + 零关节力矩：J 仍为 `0.0` —— 推扰功不计入关节正功。
- warmup 200 步后开窗：`warmup_steps=200`、`steps=40`，一步都没有混进窗口。
- 窗口停在 1347 步（= 336×4+3，非 4 的倍数）：`partial=true`，逐步与逐关节求和仍等于总量。

### 21.6 只读（E16、E17）

- AST 扫描 `energy_recorder.py`：`set_*` / `write_*` / `apply_*` / `reset_*` / `step` 调用 **0 处**。
- 跨进程开/关对照：250 个策略步 × 40 个量（root 位姿、12 关节 q 与 q̇、12 个施加力矩），
  sha256 相同、逐元素 max\|Δ\| = **0.0**。两份原始数组留档为
  `reports/energy_readonly_{on,off}.npy`，可随时重新求差。

### 21.7 本轮踩到的坑

**float32 里做"闭式值"对照会造出假失败**。第一版 E10 报 rel 5.98e-08 失败，原因是
`3.0 * qdot_float32_array` 在 numpy 里仍是 float32，`sum()` 也在 float32 里累加；
计量器本身是 float64。把对照量显式升到 float64 后 rel = 1.7e-16。
留档仍是 float32（与 v2 同精度），但**任何对照运算必须升到 float64**。

### 21.8 观察（不构成失败，留给 pilot 复核）

同一段 10 s、命令 [0.5,0,0] 的闭环里，逐关节能量分布极不对称：
RL_thigh 46.35 J、RL_calf 46.35 J、RR_thigh 37.62 J、RR_calf 56.78 J，
而前腿四个关节合计只有约 19 J（FL_calf 5.33、FR_calf 5.27…）。
映射与力矩链已由 E02/E03 逐位核对排除，所以这不是索引错位；
更可能是该策略在 0.5 m/s 下把载荷压在后腿上。**pilot 阶段需要单独确认前腿是否被"拖着走"**
（这直接影响能量口径的代表性），本次不据此下结论。

### 21.9 接线状态

`HiRAEnergyRecorder` 已经接进 `HiraLabRunner`（`runner.recorder`，默认 None，路径完全不变）：

```
runner.recorder = recorder; recorder.begin_window()
runner.policy_step()        # 内部 4 次 substep，每步写一次 record()
recorder.finish()           # 返回 float64 逐步/逐关节/总量 + float32 原始留档
```

`policy_step` 的 4 子步循环改由新方法 `substep()` 驱动（同一代码路径），
并新增 `step_known_torque()` 供"已知力矩"类实验使用——两者都不绕过 `write_effort_lab → advance`。

### 21.10 复现命令

```bash
python tests/check_energy_meter.py --headless --device cuda:0 --phase exec
python tests/check_energy_meter.py --headless --device cuda:0 --phase readonly --recorder on
python tests/check_energy_meter.py --headless --device cuda:0 --phase readonly --recorder off
python tests/check_energy_meter.py --phase merge      # 需要带 numpy 的解释器
```

## §22 HIRA-020：G1 接入门禁与控制器包冻结（16/16）

### 22.1 结论

`reports/integration_gate.json`：T-01..T-15 + B-01 共 **16 行全部通过**，`g1_passed=true`。
冻结包：

```
CaMRA/calibration/experiments/hira_go2_18500/frozen/hira_go2_18500/
├── controller.json   # 身份/关节双射/增益/观测契约/时钟/资产/能量口径/环境依赖
├── policy.pt         # TorchScript（sha256 与张量摘要双登记）
├── urdf_go2.urdf     # 来源 URDF 随包分发（自证关节顺序与限幅）
├── manifest.json     # 逐文件 sha256 + bundle_id
└── README.md
bundle_id = 3c32fe31b2753a08f83779d0ddb2b6fa433252873092d3adf484421d4a047975
```

### 22.2 T-08：预登记判据在本工程不可达，按用户裁定拆分

预登记 T-08 写的是「1 env 与 32 env 的 τ 轨迹一致（atol 1e-4）；env 之间无耦合」。实测：

| 对照 | max\|Δτ\| | 含义 |
|---|---|---|
| 解耦判定：其余 31 个 env 反向，env0 | **0.0（逐位）** | env 之间确实无耦合 |
| 1 env vs 32 env（env0） | 7.95 N·m | 预登记原判据失败 |
| 1 env 放到与 env0 相同世界坐标 vs 32 env | 12.78 N·m | 不是"位置没对齐" |
| **只有一个 env，只把出生点平移 2 m** | **12.45 N·m** | 物理等价也发散 → 差异与耦合无关 |

首步就有 ~4.5e-6 的差（float32 ULP 量级），第 4 个策略步越过 1e-4 后指数放大；
行为统计上单 env 落在批量散布的 **1.10σ** 内（vx 0.4459 vs 0.4506±0.0044）。

**结论**：原判据测的是"世界坐标相关舍入 + 接触混沌放大"，不是耦合；在 GPU PhysX 上
逐位 batch==single 不可达。已按用户裁定写入修正案
`config/integration_acceptance_amendment_t08_v1.json`（sha `037d209d4efb…`）：
T-08 拆成 **T-08a 解耦判定（逐位 0.0）** 与 **T-08b 行为等价（≤3σ）**，
原判据的失败实测与对照实验**原样保留**在 `reports/batch_isolation.json`。

附带两条诊断：`enhanced_determinism` 对同一批大小是**空操作**（1 env 默认 vs 1 env 增强：逐位相同），
因此 016–019 的结果不受影响、也无需重跑；批内逐位一致本工程不提供，采集器不得假设。

### 22.3 门禁矩阵的实现方式

`tests/check_integration_gate.py` 不是"读 verdict 抄一遍"：

- **独立复核**：T-01 现场重算 checkpoint 散列比对登记值；T-05 重算 6000/1500/300/301 的算术；
- **指名到检查项**：T-06 用 `C5-command-fidelity + N2-command-override`，T-07 用
  `C7-termination-capture + C8-per-env-reset + N3-cross-env`，不靠整体 verdict；
- **证据文件散列**：12 份报告的 sha256 全部写进门禁报告，改任何一份都会被发现；
- **T-08 行**引用修正案散列，并保留原判据的失败数字（`original_criterion_passed=false`）。

### 22.4 控制器包的三层防伪

只核散列挡不住"改完内容重算 manifest"的伪造包，所以 `load_bundle()` 有三层：

1. **散列**：逐文件 sha256 + `bundle_id`（覆盖描述内容与文件清单）；
2. **语义**：关节双射性、时钟算术自洽、观测块无缝覆盖 0:45、
   `friction == source_friction`（防 0.5/1.0 那类静默回退）、能量口径必须显式；
3. **来源自证**：源 URDF **随包分发**，加载时重解析——`source_order` 必须等于 URDF
   声明顺序、effort 限幅必须等于 URDF 值；另支持外部参照（运行时审计的 DOF 顺序）。

拒混测试 6/6（R1 位翻转 / R2 换 target 顺序并重封 / R3 改时钟并重封 /
R4 改 controller_id 并重封 / R5 摩擦写回 0.5 并重封 / R6 只改 source_order 并重封）。
R2/R6 是"散列全对的自洽伪造包"，分别被**运行时参照**和**包内 URDF** 拦下——
这正是只核散列的实现会漏掉的两类。

### 22.5 最小重载确认（B-01）

从冻结目录加载（散列+语义+URDF 自证+外部参照全过）→ 校验包内描述与将运行的代码逐项一致
（action_scale/clip/kp/kd/decimation/关节顺序/默认角）→ 用**包内** policy.pt 与 URDF 跑
10 个策略步：τ 有限、max|τ| 未越限幅、高度 0.42→正常站立、指纹已记录（`bundle_reload.json`）。

### 22.6 边界与遗留

- `usd_cache` 目录不随包分发（现场由 URDF 转换，缓存路径记录在 controller.json）；
- T-11 中 v2 训练期原始日志（TASK-024/026/036）不在本机，已登记为对照基准限制；
- 冻结目录不可变：改动需 `--refreeze` 显式声明，并产生新 controller 版本；
- 采集器（HIRA-021 起）应以 `load_bundle()` 作为唯一入口，不得绕过校验直读文件。

## §23 HIRA-021：采集通路与隔离恢复测试（21/21）

### 23.1 结论

`reports/collector_integration_tests.json`：**21 项全过**（plan 9 + exec 9 + legacy 3），
`all_passed=true, missing=[]`。采集器 `hira_go2.collector.HiraCollector` 已与 G1 冻结包接通。

### 23.2 采集通路（C01–C07，真实仿真）

- **入口唯一**：`HiraCollector` 构造即 `load_bundle()`（散列+语义+包内 URDF 自证+运行时参照），
  controller_id=bundle 目录名；本轮 prereg 与采集配置被散列写进每条 trial 的 meta
  （C01：bundle_id `3c32fe31…`，experiment `hira_go2_18500`）。
- **身份显式**（C02）：experiment/controller/bundle_id/policy_sha/run_id/split/trial_id/
  program(program_sha256)/profile/seed/attempt/attempt_kind 全部落 meta，不靠目录名推断。
- **时钟**（C03/C04）：warmup 200 步后窗口起点对齐（首配对点 t=1.000 s）；
  配对点 = 区间数+1 = 21（端点共享），命令列与写入值逐元素一致；
  50 Hz 状态 100 点。
- **能量**（C05）：`physics_200hz.npz` 留档 applied_torque/qdot（源顺序），
  离线 `integrate_series` 复算 rel = 0.0 / 4.4e-16。
- **推扰**（C06）：按 trial 时刻表施加，实测 Δv=(0.50000006, 0.0)。
- **逐 env 终止**（C09）：注入把 env0 压到地面 → 终止在 reset 前记录（t=0.60 s），
  随后该 env 的 profile 旋钮与初态被恢复；env1 的命令序列与终止记录完全不受影响。

### 23.3 恢复语义（R01–R09，纯函数直接测）

`plan_attempts()` 是唯一裁决处，对伪造目录状态验证了 9 类情形：
空 run / DONE 完整 → skip / **DONE 在但输出被篡改 → resume**（过时 DONE 不算数）/
中断（running 无 DONE）→ resume 且 attempt+1 / 业务失败 → **blocked（不自动重试）**/
部分批（缺的 run、有的不动）/ v2 命名被拒 / v2 控制器包被拒（改名重封也拦下）/
计划不重复计数。
中断补跑实测（C08）：两条 trial 的 attempt_01 伪造成 running 后 resume，
**旧目录逐字节保留**，新结果进 attempt_02，完成后 plan 判 skip——覆盖、重复计数都不会发生。

关键设计：所有写入先 `*.tmp` 再 `os.replace`；DONE 里登记逐文件 sha256；
attempt 目录只增不改；中断补跑（resume）与业务失败重试（须显式指令）是两条线。

### 23.4 新旧语料隔离

- HiRA trial_id **必须以 `hira-` 开头**（前缀白名单），v2 的命名一律拒（R07）；
  v2 控制器包即使改回 v2 id 并重封，也在构造时被拒（R08，散列全对的自洽伪造包）。
- `recompute_v2_trial.py`（在存有 v2 语料的 .10 上运行）：
  - **f32 口径复算** J=588.431751713 vs 落盘 588.431751713，**rel 2.7e-15** —— v2 数据可信；
  - **f64 口径**（HIRA 的方法）差 rel 2.57e-10 —— 记录为方法差异；
  - HiRA TrialSpec 拒装 v2 trial_id（无 `hira-` 前缀）。
- v2 校准 src/tests 的 git 工作树干净（L01），既有 CPU 核心测试全过（L02）。
  两个坑记录在案：pytest 必须以 `CaMRA/calibration` 为 cwd（`from sim.…` 依赖 conftest）；
  v2 npz 里的 `sim_dt` 是 float32（0.004999999888…），复算必须用 meta 里的 0.005，
  否则会得到 2.2e-8 的假差异。

### 23.5 遗留

- 同一批内 obs_noise / window_s / warmup_s 必须一致（obs 级开关无法逐 env）——
  022 的 20 条 pilot 分批时按 profile 分组即可。
- 采集吞吐未系统测量（022 的资源估算任务）。
- 022 起正式采集的 run 目录约定：`/pub/data/caohy/logs/camra_hira/runs/<run_id>/`。

## §24 HIRA-022：20 条五条件 pilot（10/10）与资源实测

### 24.1 结论

run 目录 `/pub/data/caohy/logs/camra_hira/runs/pilot_20261105T223428`：
4 命令族 × 5 profile = **20 条全部完成采集与验证**（`reports/pilot_validation.json`，
P01–P10 全过，`all_passed=true`）。

### 24.2 清单与可复现性

- 4 族（straight_shift / arc_combined / alternating / start_stop_turn，与 v2 同一个
  `camra_calibration.commands.build_program` 生成器——命令表可逐元素复现）×
  5 profile（clean / friction_only / mass_only / push_only / combined）；
- seed 6110..6129 为 **pilot 专属区间**，已写进 collect_config 并注明后续 split 必须避开
  （HIRA-024 登记互斥表时引用）；
- 每条 trial 的命令表显式存进 meta（program_sha256），验证时重新生成比对：P03 逐元素一致。

### 24.3 验证结果

| 项 | 实测 |
|---|---|
| DONE / 身份 | 20/20 DONE 完整；meta 的 bundle_id 全部等于 G1 冻结包 `3c32fe31…`（P02 = "正式 runner 跑的是冻结包"的证据） |
| 时钟/配对 | 301 配对点（Δt=0.1）、1500 状态点、6000 物理步，20 条全一致 |
| 能量 | 离线复算最大 rel **4.3e-15**；J 范围 444–41015 J |
| profile 旋钮 | 无条件绝对式写入（幂等）：readback == spec 值，20 条全对 |
| 推扰 | 28 次，实测 Δv 与指令差 < 0.05 m/s |
| 名义/执行配对误差 | **报告口径非判据**：全体中位 1.041 m；分组中位 clean 0.381 / friction 0.298 / mass 1.990 / push 1.490 / combined 1.509 m —— 开环跟踪（无航向闭环）的固有漂移，扰动组显著更高，符合预期 |
| 业务结果 | 完成 16 条，**提前终止 4 条**（全部是 combined/噪声组）——如实计数，不隐身、不丢弃 |
| 低速 | 程序低速段占比 40%–73%（协议下限 25%），10/20 条含 near_zero 段 |

### 24.4 本轮抓到的真 bug：跨批旋钮污染

首轮 20 条跑完后，独立验证发现 **mass_only 批继承了 friction_only 批的采样摩擦、
push_only 批带着上一批的 +3 kg 跑**：`_apply_mass` 是增量式、且非相关 profile
不复位旋钮。修复：构造时快照基准质量/惯量，`_apply_mass` 改为**绝对式**
（base+Δ，幂等），`_apply_spec_env` **无条件**写全部旋钮。被污染的首轮 run
目录已删除，修复后重跑（本 § 的数字全部来自修复后的 run）。
这类错误在 v2 里靠"每 env 独立写旋钮+读回"防住了；HiRA 版第一版漏了"跨批复位"，
被验证脚本抓到——验证独立于采集器再次证明必要。

### 24.5 资源实测（更新 025/030/032/034 的估算）

- **吞吐：4.6 s/条**（30 s 窗口，4 env 一批，含批间开销；总 91.9 s 采完 20 条）；
- **存储：1.15 MB/条**（压缩 npz；20 条共 23 MB）；
- 峰值 RSS 4.2 GB；
- 外推：baseline 400 ≈ **31 min**、calibration 800 ≈ **61 min**、heldout 400 ≈ **31 min**、
  near_obstacle 200 ≈ **15 min**，总存储 ≈ **2.0 GB**（v2 同规模为 68 GB——
  v2 逐 trial 存了全量 root/接触调试数据，HiRA 只存能量计量与配对所需字段；
  若 036 审计需要更多字段，需在此之前扩记录器并重新过受影响门禁）。

### 24.6 遗留

- combined 组 4 条摔倒：噪声条件下策略跟踪失败是**业务结果**，已记录；
  024 生成正式清单时噪声格的配额仍按协议，不因 pilot 表现调格；
- P08 的配对误差只是漂移分布；真正的"跟踪质量"统计（含名义偏差分位数）
  在 028 的消费接口里按协议口径计算。

## §25 HIRA-023：HiRA 专属校准协议与功率选型冻结

### 25.1 产出与散列

| 文件 | sha256（前 16） |
|---|---|
| `config/calibration_protocol_hira_v1.json` | `b5069a3721ea1054` |
| `config/power_model_prereg_hira_v1.json` | `41a3cc2ea5791e9f` |
| `reports/protocol_diff_from_v2.json` | 逐项差异报告（9 项，`user_rulings_required=[]`） |

**继承校验**：`calibration_protocol_v1.0.2.json` 与 `power_model_prereg_v1.json`
的散列在生成前后逐一比对，**未被改动**。

### 25.2 继承与覆盖

逐字段继承 v1.0.2：时钟（30 s / 200/50/10 Hz / 301 配对点）、命令域（±1.0/±0.4/±1.0）、
四族五条件、J 正功契约与 J_min=1 J、七分片 1820（20/400[240/80/80]/800/400/200）、
Q0.9 nearest-rank、bootstrap 10000×95%、簇与分母规则、每格完整率 ≥0.9、
non_computable>0.20 暂停、三类无效规则、近障地图构造规则与"不得按新 δ 重设计"。

覆盖项（全部为身份/簿记，无统计语义变更，`user_rulings_required` 为空）：

1. **身份绑定**：applies_to → G1 冻结包（bundle_id `3c32fe31…`、manifest/controller/policy/urdf 散列）。
2. **种子块**：baseline_fit/dev/validate、heldout、near_obstacle 与 v2 **同块**
   （同 seed + 同生成器 = 同命令程序，供 HIRA-024 跨控制器匹配）；calibration 改为
   6130–6929 以避开 HiRA pilot 已占的 6110–6129；七块互斥、reserved 排除保持。
3. **trial_id**：强制 `hira-` 前缀（采集器白名单）。
4. **能量测量实现登记**：τ=applied_torque、子步结束相位、**float64 乘法**
   （v2 为 float32 乘法，系统差 ~2.6e-10 相对，已实测；跨控制器 J 比较须注明口径）。
5. **attempt_policy 显式化**：第一有效主 attempt；业务失败不补成功替换；
   基础设施重试保留原 attempt；v2 的 D-007（baseline_validate +31）/D-013（near +2）
   两次整批重采登记为对照口径差异（按最大 attempt 去重后 v2 有效数不变）。
6. **近障归属**：CAL-021/CAL-023 → HIRA-027/HIRA-029；冻结时点 = HIRA-029 数据锁。

### 25.3 一处向继承语义的对齐（不是新决策）

v1.0.2 的 `energy_partial_rule`：env 终止即该 trial 结束，**J 只覆盖实际执行步数，
不得外推**。HiRA 采集器首轮实现为全窗口累计（含 reset 后的挣扎段）——违背继承语义。
已修正：`stop_substep` 逐 env 记录、输出按截断段切片、`energy.json` 按截断段
float64 复算、outcome 对齐 v2 口径（终止 = `business_failure`/reason=terminated）。

效果在 pilot 重跑中直接可见：J 上界从 41015 J（挣扎段累积的病态值）回落到
**1223 J**；combined 组配对误差中位从 1.509 m 降到 0.426 m（漂移尾巴随截断消失）。

### 25.4 协议口径下的 pilot 重跑（pilot_20261106_protocolv1）

P01–P10 全过；20/20 DONE；J 范围 365.9–1223.3 J；16 完成 + 4 终止（全为噪声组，
如实计数）；4.2 s/条、0.94 MB/条；外推 400/800/400/200 ≈ 28/56/28/14 分钟、1.6 GB。
首轮 run（全窗口口径）保留在 `pilot_20261105T223428` 作对照。

### 25.5 冻结时序

协议冻结时间戳早于任何 baseline_fit 采集与拟合、早于一切正式结果查看；
pilot 的角色是链路验证（`may_tune=false`），其产物不进入拟合或正式统计。

## §26 HIRA-024：七分片清单与跨控制器匹配表

### 26.1 产出

| 文件 | 说明 |
|---|---|
| `manifests/{pilot,baseline_fit,baseline_dev,baseline_validate,calibration,heldout,near_obstacle}.jsonl` | 七个分片清单，1820 条 |
| `manifests/lock.json` | 逐 split 条数 + 文件散列 + 协议/生成器绑定 + 消费策略 |
| `reports/split_independence.json` | 21 组两两交集、格级低速、pilot 绑定 |
| `reports/cross_controller_match.json` | 逐条 `source_trial_id` 与未匹配原因分解 |
| `reports/manifest_validation.json` | 独立验证 8 项（M01–M08，全过） |

清单是 **(协议, 生成器)** 的纯函数：直接复用 v2 的
`camra_calibration.trials.build_all(hi ra_protocol)`，写出时只加 `hira-` 前缀。
M03 用同一函数重跑并逐条比对 trial_id/seed/两个散列——六个生成 split 全部复现。

> 注意两个"协议 sha"：文件散列 `79f983f8…`（`Protocol.load` 用，配套 `.sha256` sidecar）
> 与内容自散列 `b5069a37…`（文件内 `sha256` 字段）。两者不是同一个东西，引用时要写清。

### 26.2 配额与结构

| split | 条数 | 格数 | 匹配 v2 |
|---|---|---|---|
| pilot | 20 | 20 | 5 |
| baseline_fit | 240 | 4（clean only） | **240** |
| baseline_dev | 80 | 4 | **80** |
| baseline_validate | 80 | 4 | **80** |
| calibration | 800 | 20 | 280 |
| heldout | 400 | 20 | **400** |
| near_obstacle | 200 | 20 | **200** |
| 合计 | **1820** | — | 1305 |

格级低速占比最小 0.444（协议下限 0.25）；七 split 的 seed / 模板 / 簇 / trial_id
两两交集**全部为 0**（21 组）。

### 26.3 两个散列，各司其职

| 字段 | 口径 | 用途 |
|---|---|---|
| `program_sha256` | 只哈希命令表（与采集器 `TrialSpec.program_sha256` **完全同口径**） | 清单 ↔ `meta.json` 绑定校验 |
| `template_sha256` | `Program.sha256()`（含 family/seed/参数/命令表） | 跨控制器匹配键（v2 清单的 `program_sha256` 是同一口径） |
| `cluster_id` | `family:template_sha256[:16]` | 簇标识 |

这条是踩出来的：首版清单只记 `Program.sha256()`，与 pilot 的 meta 对不上（采集器只哈希
命令表）。补齐双散列后 pilot 20 条的 `program_sha256` 全部可由 (family, seed) 重建。

**簇的语义**：生成器为每个 (family, seed) 采样独立目标骨架，所以一条 trial 自成一簇
（簇数 = trial 数）。这与协议"bootstrap 以整条轨迹为最小单位"一致；报告里不再另设簇内样本。

### 26.4 跨控制器匹配

按 `template_sha256` 逐字相等匹配（同 seed + 同生成器 = 同命令程序），**不做近似匹配**：

- baseline（240/80/80）、heldout（400）、near_obstacle（200）**100% 配对** —— 块起点与 v2 相同；
- calibration 280/800（35%），未匹配 520 条分解为：
  **130 条** seed 不在 v2 块内（6800–6929），**390 条** seed 相同但 v2 分给了别的族
  （块起点差 130，130 mod 40 = 10 → 格边界错位）；
- pilot 5/20（pilot 是既成事实，不做匹配优化）。

未匹配的 520 条不是缺陷：HIRA-037 的比较口径是"同分布非配对对照"，
配对只是额外便利（baseline/heldout/near 已 100% 可用）。

清单逐条保存 `source_trial_id`、`source_split`、`source_match`，可独立重建匹配关系。

### 26.5 结果区隔离

`manifests/lock.json` 的 `consumption_policy` + 可执行的 `results_guard.py`：

- 受保护 split：`heldout`、`near_obstacle`；解锁条件 = `config/margins_hira_v1.json`
  存在且 `state=frozen` 且含 `alpha`/`delta`（HIRA-031 产出）；
- 采集/分析/publish 前必须 `assert_allowed(split, stage, EXP)`；不接受环境变量绕过；
- M07 实测：当前 heldout 被拒（"缺少 margins_hira_v1.json"）、calibration 放行。

### 26.6 遗留

- calibration 的 390 条"同 seed 异族"未匹配无法通过改清单修复（种子块已在协议里冻结）；
  若 037 需要更大配对样本，只能按"同分布非配对"处理并如实说明配对率。
- 清单本身不含 profile 采样值（摩擦/质量/推扰），那些由 `sample_spec(seed)` 在采集时
  确定性生成并落 meta；037 若要做配对比较，须从两边 meta 读取实际 profile 读回值。

---

## §27 HIRA-025：400 条 clean 功率基线（fit/dev/validate）

**目标**：按已锁定的 manifest 采集 HiRA 专属 clean 基线，逐条资格核对，不完整条目进排除明细。

**入口**：`tests/run_split.py`（通用 split 采集器）——manifest 是唯一 trial 来源，命令表用
`camra_calibration.commands.build_program` 重建后与清单的 `program_sha256`/`template_sha256` **双散列对账**，
任何一条对不上就拒绝开跑（不静默）。按 profile 分批（同批 obs_noise/window/warmup 必须一致），
断点续跑（DONE 完整则 skip，不足一批用 scratch 占位填满 env 槽位、跑完即删）。

**结果**（run 目录 `/pub/data/caohy/logs/camra_hira/runs/baseline_20261106_{fit,dev,validate}`）：

| subset | 条数 | 完成 | business_failure | 墙钟 | 目录 |
|---|---|---|---|---|---|
| baseline_fit | 240 | 239 | 1（straight_shift-0005，5336 步） | 10.6 min | 272 MB |
| baseline_dev | 80 | 79 | 1（alternating-0007，5276 步） | 4.0 min | 91 MB |
| baseline_validate | 80 | 80 | 0 | 3.8 min | 91 MB |

* 单条 2.05–2.30 s（`--num-envs 8`），**1.135 MB/条**，400 条共 454 MB。
* 完整性核对 `tests/check_baseline.py` **10/10 全过**、400 条有效、**0 条排除**：bundle_id/controller/seed/
  program_sha256 逐条与清单及冻结包一致；clean 旋钮读回 friction=1.0、mass 增量 0；energy 离线复算
  **最差 rel = 0.0**（400 条）；每格完整率 0.95–1.00（下限 0.9）；簇与 trial 一一对应、簇前缀=族、族簇数=配额；
  seed 与 pilot 区间无交集；每个 split 有独立 run 目录且 `collect_config` 的 split/manifest/散列/bundle 自洽。
* **validate 保持封存**：核对阶段只登记 outcome/步数/点数/能量复算残差，`j_exec` 字段显式置空，
  不计算任何拟合误差（拟合误差在 §28 冻结模型之后才第一次算出）。

**修掉的两个真问题**

1. **终止 trial 的配对点没有按时间截断**。`state50` 按控制步（4 物理步/点）、`pairing` 按命令区间
   （20 物理步/点）采样，早先两者都用 `n_stop//4+1` 切，切片长度超过实际点数而不生效，终止 trial 会留下
   **终止之后（reset 后挣扎段）** 的配对点。首轮被 B04 抓出（5336 步却报 301 个配对点）。
   修正为 `n_stop // steps_per_interval + 1`，重跑后 B04 通过。
2. **三个 split 共用一个 run 目录**会让 `collect_config.json` 只记录第一个 split（fit），后两个 subset 的
   meta 都指向错误的配置。改为每个 split 独立 run_id（`baseline_20261106_{fit,dev,validate}`），
   并新增 B09 判据专门抓这一类"目录被多 split 复用"。

**环境事故（本轮耗时主因）**：`/home`（autofs → NFS）在 11:07 整体挂死（两台机器同时，`ls` 超时），
采集进程会卡在 AppLauncher 之后——它在读 NFS 上的 frozen bundle，不是代码问题。处置：把 adapters /
calibration src / 实验目录逐文件复制到本地盘 `/pub/data/caohy/work/hira`，从副本运行；守护脚本
`hira_autoprep2.sh` 负责"NFS 恢复→复制→校验 13 个关键文件→自动开跑"。
另外：本地 Bash 工具的 stdin 不可用，`ssh "cat > f" < local` 只会得到 **0 字节**文件；
`scp` 在 NFS 卡住时也会挂半途。可靠传输方式是把内容 base64 后作为 ssh 参数传。

---

## §28 HIRA-026：拟合、冻结并独立验证 HiRA 名义功率模型

**口径**：`config/power_model_prereg_hira_v1.json`（sha `d7a56d273d6524c1…`，state=frozen，早于任何采集冻结）。
特征只用命令的 `|vx|,|vy|,|wz|` 及其平方/乘积（恒非负），拟合为 relative-weighted NNLS（权重 1/J_exec，
`scipy.optimize.nnls` + 列 L2 归一化回代），目标 `J_nom = Σ_{i=0..299} P̂(u_i..u_{i−K})·0.1`。

**数据**：fit 239 条（1 条 business_failure 按 `outcome_not_completed` 排除）、dev 79 条（同）、
validate 80 条（0 排除）。每条都通过 10 类排除规则检查（目录/DONE/outcome/步数/program 散列/命令表逐点
≤1e-5/J_exec 反算 ≤1e-9/前缀/bundle 绑定/pairing 形状）。

**候选与选型**（只用 fit 训练、dev 选型）：

| 候选 | 特征 | n_params | fit q90 | dev q90 | 截距份额 |
|---|---|---|---|---|---|
| C1_lin_k0 | 1,\|vx\|,\|vy\|,\|wz\| | 4 | 6.43% | 5.64% | 0.452 |
| **C2_quad_k0** | +vx²,vy²,wz²,\|vx\|\|wz\|,\|vy\|\|wz\|,\|vx\|\|vy\| | **10** | **4.41%** | **4.35%** | **0.470** |
| C3_quad_k1 | C2 + 1 条历史命令块 | 20 | 4.41% | 4.35% | 0.470 |
| C4_lin_k2 | C1 + 2 条历史命令块 | 12 | 6.53% | 5.65% | 0.453 |

C3 与 C2 的 dev q90 在 1e-4 内相同（历史块与当期块高度共线，NNLS 只是把权重挪到 lag 列：
lag1 的 `abs_vy`=2.336、`wz2`=6.776 非零），按预登记 tie 规则（±0.005 内取参数少者）**选 C2_quad_k0**。

**冻结**：`config/power_model_hira_v1.json`，模型 sha **`27bc5fe1423256ce…`**（散列只覆盖 content_dict，
不含时间戳；重跑一次得到同一 sha，证明拟合可复现）。权重：
`[10.4148, 4.6562, 2.2988, 0.0, 39.7912, 47.0641, 6.7762, 0.0, 3.4692, 5.5223]`
（`|wz|` 与 `|vx||wz|` 被压到 0：本策略的转向功率主要体现在 `wz²` 与 `|vy||wz|`）。

**validate 一次性结果**（模型落盘之后才算）：q90 **5.88%**、中位 1.95%、最大 13.24%；
带符号偏差 q90 +4.88%、中位 +0.61%（轻微高估）；0 条 non-computable、0 条越出输入域。分组：
`start_stop_turn` 7.27% > `straight_shift` 4.73% > `arc_combined` 4.22% > `alternating` 3.23%；
低速组 7.14% vs 高速 4.45%；平稳段 7.27% vs 过渡段 4.40%。准入 4/4（dev ≤0.10、validate ≤0.10、
构造性非负、non-computable 份额 ≤0.2）。

**12→13 项自证**：前缀因果性（把命令 `[151:]` 换成随机值，`[0:150]` 的 P̂ **逐位不变** = 0.0，
而后缀确实变了 78.66 W，证明篡改生效）；非负（三 split 全部区间 `min P̂ ≥ 0`、截断份额 0）；
**由落盘文件重算**：从 JSON `load(verify_hash=True)` 得到的模型与**拟合进程内存模型**算的 J_nom
**逐位相等**（30 条），另一条求和路径（Φ·w 聚合）差 2.3e-15 相对（ULP 级，仅作诊断）。

**第一次判定为 FAIL 的原因**：我最初把"另一种求和顺序"当成"独立复算"，用绝对 1e-12 判，2e-15 相对
（约 2e-12 J）就超了。已改为"同一实现下文件模型 vs 内存模型逐位相等"作主判据，另一路径按相对 1e-12 判。
第一次的报告保留为 `reports/power_model_validate_run1.json`（同一模型 sha、同一 validate q90，未改选型）。

**K>0 约定**：选中模型 `history_k=0`，故"reset 后 zero_history / 上层增广命令历史"约定**不适用**；
C3/C4 的 lag 语义（越界钳到首点 `u_0`，等价 zero-order hold 而非零命令）已在 `feature_matrix` 中固定，
若将来改用 K>0 模型必须沿用该约定，否则第一个区间的 P̂ 会变。

**与旧模型的隔离**：新模型只依赖命令，不引用任何执行侧量；文件为 `config/power_model_hira_v1.json`，
与 v2 的 `power_model_v1.json` 无交集（后者未被读取、未被修改）。

---

## §29 HIRA-027：重验 HiRA 执行系统的 Go2 占地轮廓

**问题**：v2 的轮廓（0.889 × 0.585 m）是从另一份 USD 资产算的，不能直接沿用；必须从本候选
自己的碰撞几何重算，并且只用已采到的独立数据定姿态包络。

**资产几何**（`config/go2_asset_geometry_hira_v1.json`，sha `efcc590ff228fc92…`）：
由 `tests/extract_geometry_hira.py` 直接从包内的 `urdf_go2.urdf` 解析 —— 29 links / 28 joints /
27 colliders（`cube` / `cylinder` / `sphere`，无 mesh）。schema 与 v2 的 `camra-go2-asset-geometry-v1`
一致，因此可直接被 `AssetGeometry.load(verify_hash=True)` 读取。两处转换约定：
URDF 的 `origin` 就是 child 相对 parent 的零位姿（`pos1=0`、`rot1=identity`）；
URDF 的长方体三边不等，而采样器只支持立方体，故用单位立方体 + `matrix_link_from_shape` 的对角缩放。
`rest_consistency`：FK(全关节=0) 与资产作者的链接静止平移最大差 **2.2e-13 m**。

**姿态包络**：623,757 帧 / 420 条 trial（fit 359,834 + dev 119,819 + validate 120,000 + pilot 24,104），
全部来自 clean baseline 与 pilot 的 50 Hz 关节记录；**没有**用 heldout / near_obstacle（被 results_guard 封锁）。

**结果**：轮廓 52 个顶点、面积 **0.6769 m²**、包围盒 **1.0164 × 0.7810 m**（对 v2 的 1.143 × 1.335 倍）。
覆盖检查把同一批 623,757 帧的**全部碰撞体投影点**逐点判内：`max_outside = -0.0`、越界点 **0**。
轮廓冻结为 `config/footprint_hira_v1.json`（sha `ed23b46e46af23d2…`，含 sha256 侧车）。8/8 检查通过。

**一个必须写下来的否定结论**：我原本假设"关节限位角点组合"（每腿 3 关节取 8 角点、腿间独立 → 4096 帧）
会给出实测包络的**保守超集**。实测相反：它在 x/y 方向更长（1.067× 与 1.114×），但**面积只有实测的 0.8786 倍**
——因为角点组合缺少"单腿前伸/后蹬"这类**不对称**姿态。所以它只能当量级参照，不是兜底轮廓；
若将来实测姿态超出包络，必须重新采集数据并重算轮廓，**不得**用角点轮廓顶替，也不得靠放大 δ 掩盖。

**性能教训**：`coverage` 对 62 万帧逐点算多边形距离要 **~17 分钟**（outline 3 分钟）。
脚本加了 `--reuse`：多边形从已冻结的 footprint 文件读、coverage/limit 从旧报告读，只重做判定（秒级）。
判据写错时不必重算 22 分钟。

---

## §30 HIRA-028：能量步成本与未来导航消费接口

**交付**：`config/step_cost_schema_hira_v1.json`（schema 散列 `a52563f95818f659…`，文件散列
`1c6b50403cb8cb5d…`，+ `.sha256` 侧车）与 `reports/step_cost_interface_tests.json`（13/13）。
接口直接复用 v2 的 `step_cost.py`（`c_nom = P̂·0.1 s`、`c = (1+α)·c_nom`、`BudgetState`、
`StepCost.step`、`trajectory_cost`、`energy_ratio`），只把模型换成 HiRA 的冻结模型。

**α 的处理**：α 要到 HIRA-031 才冻结，所以测试里用的 α=0.25 在报告中逐处标注
`alpha_is_test_value=true`，并写明"不得被下游当作标定结果引用"。

**13 项结果**

| id | 结论 |
|---|---|
| S01 | 100 条轨迹（validate 80 + fit 20）的 300 个区间成本之和 = 整窗 J_nom，最差 rel **4.64e-15** |
| S02 | 每步 `c = (1+α)·c_nom`；Σ 比值 **1.250000**（α=0.25） |
| S03 | 预算 spent/remaining/exhausted/z_ratio 自洽且剩余单调不增 |
| S04 | 零/负/NaN 预算、旧 `a.u.` 单位、非 `mechanical_J` 口径全部被拒 |
| S05 | 提前终止只累计实际执行 137 步（385.60 J < 整窗 875.45 J），不外推不补齐 |
| S06 | 权重改 1e-6 后 `load(verify_hash)` 拒绝；正品可载 |
| S07 | `dt_s=0.05` 与模型 `step_cost_dt=0.1` 不一致时构造即拒 |
| S08 | K=1 时 `step` 与 `feature_matrix` 逐位一致（0.034000）、zero_history 首步一致、require_explicit 缺历史报错 |
| S09 | 交付模型 K=0 → 上层状态无需增广（`none: command is the full Markov input`） |
| S10 | schema 自洽：dt=0.1、300×dt=30 s、单位 J、绑定模型散列 |
| S11 | 20×300 步中负功率/负成本 0 个 |
| S12 | 步成本绑模型散列、足迹绑几何散列，两者引用不同对象；δ 只在几何侧施加一次 |
| S13 | ε_c=(J_exec−J_nom)/J_nom；J_nom<J_min 记 `non_computable` |

**两处接口语义（下游必须知道）**

1. **`remaining_J` 允许为负**：S03 里 174 步把 500 J 预算用到 spent=500.2356、remaining=**−0.2356**。
   `z_ratio` 会夹到 [0,1]（实测 0.0），但 `remaining_J` 保留负值以表示**超支量**。
   下游若用 remaining 做阈值判断，需自行 `max(0, ·)` 或直接用 `z_ratio`。
2. **`zero_history` 会覆盖 reset 传入的历史**：episode 第一条命令被当作它自己的历史（与
   `feature_matrix` 的"越界钳到首点"一致）。要验证"外部历史参与预测"必须 `reset_rule="require_explicit"`
   并显式 `reset(history=...)` —— 我第一版 S08 就是因为没注意到这点而误判为失败（0.032 vs 0.034，
   差的正是"历史块用了 u 还是 h"）。

**边界**：本任务只交付可消费接口，没有启动任何导航训练；不把 J 预算与旧论文的 a.u. 预算混用，
也不默认继承旧 α。

## §31 HIRA-029：正式采集门禁 G2（13/13，锁 v1 → v2）

**这一节解决什么**：校准数据能不能开跑，不再靠"我觉得准备好了"。锁把"用哪个模型、哪个轮廓、哪份清单、
哪一版代码、哪个地图"全部固化成一份可核验的 JSON，采集器在**启动仿真之前**核对，任何一项不符直接拒绝。

**锁里有什么**（`config/data_lock_hira_calibration_v1.json`，sha `85ea6539…`）

| 门 | 内容 | 散列前缀 |
|----|------|---------|
| controller | 冻结包与 bundle_id | `3c32fe31…` |
| protocol | 校准协议 v1 | `79f983f8` |
| power_model | 名义功率模型 | `4c3fc567` |
| footprint | 占地轮廓 | `51867346` |
| step_cost_schema | 能量步成本 schema | `1c6b5040` |
| trial_lists | 七分片 1820 条 + lock | `0a76ae09` |
| near_obstacle_maps | v2 原始物理地图只读副本 | `78c0145f` |
| code | 37 个代码文件逐个散列 | — |
| gates | 6 项前置门禁报告与其散列 | 全 passed |
| margins | `not_yet_produced` + `gated_splits=[heldout, near_obstacle]` | — |

**13 项负测试**（`tests/check_data_lock_hira.py`，每项都先复制锁到 tmp 再篡改）

| 编号 | 篡改/场景 | 期望 |
|------|-----------|------|
| L01 | 正品锁放行 calibration | 放行 |
| L02 | 锁内容改了、侧车没改 | 拒绝加载 |
| L03 | 功率模型散列错（侧车同步重算） | verify_artifacts 报错 |
| L04 | 足迹散列错 | 同上 |
| L05 | 清单散列错 | 同上 |
| L06 | 代码散列错 | 同上 |
| L07/L08 | margin 未产出时采 heldout / near_obstacle | 拒绝 |
| L09 | split 拼错 `calibraton` | 拒绝 |
| L10 | 对正式 split 用 `allow_overwrite` | 拒绝 |
| L11 | 锁文件缺失 | 拒绝 |
| L12 | 近障地图被改 | 拒绝（实测与 v2 原件同 sha） |
| L13 | 配额/磁盘/并发复核 | 放行 800 条 |

**v1 → v2 的由来**：HIRA-030 采集中发现观测噪声缺陷（`build_noise_vector` 漏乘 `DOF_VEL_SCALE`），
改的是 `observation.py` 一个文件。按"代码改了就必须新版本新锁"，重建为 `data_lock_hira_calibration_v2.json`
（sha `c229c417…`，stage `calibration_collection_v2`，`amendment` 写明 160 条 combined 的归属与重采原因，
`parent_lock` 指回 v1 保原子归属）。37 个代码文件里只有 `observation.py` 的散列变了，其余 36 个不变。
v2 下重跑门禁仍是 13/13（`reports/formal_gate_v2_lock.json`）。

**边界（下游必须知道）**

1. **锁里存的是绝对路径**，而且指向 `/pub/data/caohy/work/hira` 这个采集实际发生的工作副本。换机复核要么按同一
   布局恢复，要么在交付阶段改造成"仓库相对路径 + 仓库根"的锁格式。这是已知待处理项，不是疏忽。
2. **锁必须在代码定稿之后生成**。我先给 `run_split.py` 加 `--lock` 再生成锁，否则自己的锁当场失效 —— 这正是它该有的行为。
3. **`margins` 永远是 `not_yet_produced`，不许回写**。HIRA-031 只能追加新的 stage lock，
   让"校准数据永远归属采集时那把锁"这件事在文件层面成立。

## §32 HIRA-030：800 条正式校准与完整性审查（771/800，两格资格缺口已裁定）

**采集**：`calibration_20261106`，4 族 × 5 profile × 40 条，`num_envs=8`，单条 2.09–2.19 s，run 目录 886 MB。
入口只有 `run_split.py --split calibration --lock <v2锁>`。

**一次真事故：观测噪声大 20 倍**

源里 `noise_vec` 在 `obs_scales.dof_vel(0.05)` 缩放**之后**才进观测；我的 `build_noise_vector` 漏乘这一步，
于是 dof_vel 噪声是 1.5 而应为 0.075。combined 是唯一开噪的 profile，160 条**全部在起始数步内摔倒**
（单条只剩 ~58 KB，正是"开局就倒"的指纹）。处理：160 条零删除移入
`quarantine/calibration_v1_noise_bug`，重建 v2 锁后在原清单重采同一批 trial_id/seed（attempt 递增），
351.5 s / 20 批 / 907.2 MB。其余 640 条 `obs_noise=False`，噪声分支从未触发，数值不受影响。

**完整性 8 项**（`reports/calibration_collection_integrity.json`）

| 编号 | 判据 | 结果 |
|------|------|------|
| C01 | 800 条 / 20 格 / 每格 40 | 通过 |
| C02 | bundle/split/seed/program 散列逐条一致 | 0 不合格 |
| C03 | 五种 profile 实际旋钮 = 按 seed 重建的期望 | 0 异常；推扰 1123 次，\|Δv\| ∈ [0.007, 0.694] m/s |
| C04 | completed=6000/1500/301，失败截断自洽 | 0 不合格 |
| C05 | 能量离线复算 ≤ 1e-9 | 800 条最差 rel **0.00e+00** |
| C06 | 每格完整率 ≥ 0.9 | **未达标 2 格**（见下） |
| C07 | non_computable ≤ 0.2 | 2/800 = 0.0025 |
| C08 | 未打开 heldout / near_obstacle | 通过 |

**两格未达标的成因（逐条读记录，不猜）**

- `arc_combined/mass_only` **33/40 = 0.825**：7 条失败全部发生在 **2.10–17.76 s 的行走中段**，
  且集中在较大附加质量一档（8.8–9.9 kg 占多数）。这是**稳态能力边界**：载荷 + 转弯叠加。
- `straight_shift/friction_only` **35/40 = 0.875**：5 条里 **4 条在 ≤0.44 s**。其中 `0000` 与 `0006`
  的窗口首帧已经 z = 0.097 / 0.145 m（低于 `FALL_HEIGHT=0.15`）、倾角 168°/139° ——
  也就是 1 s 零动作 warmup 结束时人已经倒了，而这正是全部 2 条 `non_computable`。
  只有 `0003`（μ=0.323，19.5 s 才倒）属稳态低摩擦失稳。
  **注意失败与 μ 不单调**（μ=0.309 那条反而跑满 30 s），所以"低摩擦能力边界"不是主因，
  主因是 **warmup 瞬态**：1 s 零动作不足以保证所有 seed 站稳。

**裁定**（用户 2026-11-07 决定）：两格都不补采、不改 40 条/格配额、不删不替换失败记录；
按协议 `min_complete_share_rule` 作"协议裁定"而非"回到采集"，两格不进入**格级**正式结论。
α/δ 仍用全部 771 条完整记录计算（协议规定业务失败保留在主分母但作为未覆盖样本，ε 分位只用完整记录），
必须并列报告剔除两格（703 条）的敏感性结果。

**边界**

1. **warmup 缺陷是真的**，但本版协议内不修：修了与已采的 800 条不可比。记入下一版本协议建议。
   推论：HIRA-032 的 heldout 用同一套 warmup，预期同样有 ~0.25% 的"开局就倒"，届时按同样口径归类，不许临时改。
2. **不许用"追加几颗种子把格子凑到 0.9"**：那会改 C01 的配额口径、要 v3 锁，本质仍是凑通过。

## §33 HIRA-031：冻结 α 与 δ（α=0.3516，δ=9.0322 m）

**一句话**：HiRA 底层在 30 s 开环、受扰动时漂得比 v2 的控制器远得多 —— δ 9.03 m 对 2.61 m（3.46×），
α 0.352 对 0.190（1.85×）。这不是实现 bug：clean 档 δ 只有 1.85 m，膨胀全部来自扰动档。

**口径**（严格照协议 statistics 段）

| 项 | 规则 | 实现处 |
|----|------|--------|
| 分位 | nearest-rank，第 ceil(0.9·n) 个，不插值，等于阈值算覆盖 | `margins.nearest_rank` |
| α | `max(0, Q0.90(ε_c))`，`ε_c=(J_exec−J_nom)/J_nom` | `margins.build_margins` |
| δ | `Q0.90(ε_d)`，10 Hz 配对网格上执行与名义位置的**最大平面欧氏距离** | 同上 |
| 负超支 | 原值参与分位，只在形成裕度时取 max(0,·)（283 条负值保留） | 同上 |
| 区间 | bootstrap 10000×95%，**整条轨迹 / 整簇**重采样，绝不按时间点伪增 | `bootstrap_quantile` |
| 分母 | 预登记 800 与"完整且可算" 771 两套都报 | `denominators` |

**冻结值**

| 量 | 点估计 | 95% 区间（簇 bootstrap） |
|----|--------|--------------------------|
| α | **0.351568** | 0.261318 – 0.458648 |
| δ | **9.032231 m** | 5.889766 – 15.822817 |

文件散列 `57d05222`（文件）/ `e8e539c3`（内容）；派生检验锁 `data_lock_hira_evaluation_v1.json` sha `289e4137`，
`parent_lock` 指回采集锁 v2（`c229c417`），`margins.state=frozen`，**不回写**采集锁。

**分母与分类**（这批数据有两套分类，必须一起看）

| 类别 | 条数 | 说明 |
|------|------|------|
| 清单 / 可用 | 800 / 800 | 0 条数据排除 |
| 完整且可算 | 771 | 进 ε 分位 |
| prefix_only | 29 | 业务失败，按实际走完窗口截断，**留在主分母**、不进 ε 分位 |
| non_computable（完整记录内） | 0 | `J_nom<1 J` 的 2 条本身就是业务失败，故归入 prefix_only |
| 覆盖率 | 0.9637 | 771/800 |

**δ 的构成（为什么这么大）**

| profile | n | δ q90 (m) |
|---------|---|-----------|
| clean | 160 | 1.852 |
| friction_only | 155 | 3.589 |
| mass_only | 145 | 12.131 |
| push_only | 160 | 12.394 |
| combined | 151 | 12.499 |

格内最高 `arc_combined/mass_only` 22.23 m。低速占比 ≥0.3 的组 δ q90 = 6.755（n=309），<0.3 的组 12.131（n=462）：
**命令越"快"、扰动越大，开环漂移越大**，这是开环评估口径的固有后果，不是地图或几何的问题。

50 Hz 敏感性：δ q90 与 10 Hz **完全相同**（9.032231），两者最大差 0.0059 m —— 这个量不是采样栅格造成的。

**剔除两格的敏感性**（HIRA-030 裁定的 `arc_combined/mass_only` 与 `straight_shift/friction_only`，n=703）

α = 0.365083（+0.013515）、δ = 8.330097（−0.702134，−7.8%）。剔除让 δ 变小但 α 变大，方向不一致，
所以**冻结值仍取全样本**，敏感性只作诊断并列报告。

**R-07：门禁没有校验 margin 绑定**

`enforce_collection` 对 `heldout` / `near_obstacle` 只检查"`margins_path` 这个文件存在"，不校验锁里
`margins.state` 与 `margins.sha256`。想补就得改 `camra_calibration/lock.py`，而它在采集锁 v2 与检验锁 v1 的
`code` 散列里 —— 改了会让已采 800 条的归属锁验签失败，代价大于收益。处理办法：

1. 新建 `tests/check_margin_lock_binding.py`（10/10）：stage、`margins.state`、文件散列、内容散列、α/δ 逐位、
   parent 链、heldout/near 正测放行、采集锁仍是 `not_yet_produced`，并把缺口本身作为 B09 复现登记。
2. **heldout / near_obstacle 开跑前必须先跑它**，只准用检验锁。
3. 下一版协议改 `lock.py` 后，B09 应从"复现缺口"转为"负测拒绝"。

**给 HIRA-035 的前置判断（重要）**

δ = 9.03 m 是危险集扩张量，只施加一次（`obstacle ⊕ footprint ⊕ disc(δ)`）。在现有近障地图尺度下，
这个量级很可能把可行走廊整个吃掉，使近障任务在几何上不可做。按协议"大 δ 可能是真实结果，
若不可用于预定任务则记录并询问，**不通过改分位缩小**"——所以 HIRA-035 采集前先做一次只读可行性核算
（地图尺寸、走廊宽度、扩张后可行面积占比），结果拿到再决定，不先跑数据再解释。


## §34 HIRA-032：400 条 heldout 独立检验（377/400，三格资格缺口已裁定）

**这一节解决什么**：α/δ 在 §33 冻结之后，用从未参与拟合、选型与校准的 400 条数据，检查采集本身是否完整、
身份与资格是否成立。**覆盖率统计不在本节**，在 §35（HIRA-033）；本节只做"数据能不能用"，不做"裕度管不管用"。

**入口与口径**：唯一入口 `tests/run_split.py --split heldout --lock config/data_lock_hira_evaluation_v1.json`
（检验 stage lock，`margins.state=frozen`，不回写采集锁）。run `heldout_20261106`，seed 7000–7399，
20 格 × 20 条，1 s 零动作 warmup + 30 s 窗口，`num_envs=8`，GPU0。清单逐条按 `program_sha256` /
`template_sha256` 双散列重建对账。

**采集**：50 批、wall **2697.8 s**、**448.7 MB**、单条 6.5–6.9 s（GPU0 与另一用户共存；空卡约 2.1 s）。
400 个 trial 全部 `attempt_01`、**0 次重采**，收尾 `SPLIT_HELDOUT_RUN_OK` + `rc=0`。

**完整性 10 项**（`reports/heldout_collection_integrity.json`）

| 编号 | 判据 | 结果 |
|---|---|---|
| C01 | 400 条 / 20 格 / 每格 20（分母取清单） | 通过 |
| C02 | bundle/split/seed/program 散列逐条一致 | 0 不合格 |
| C03 | 五种 profile 旋钮 = 按 seed 重建的期望 | 0 异常；推扰 559 次，Δv ∈ [0.016, 0.671] m/s |
| C04 | completed=6000/1500/301、失败截断自洽 | 0 不合格 |
| C05 | 能量离线复算 ≤ 1e-9 | 400 条最差 rel **0.00e+00** |
| **C06** | 每格完整率 ≥ 0.9 | **未达标 3 格** |
| C07 | non_computable ≤ 0.2 | 0/400 = 0.0000 |
| C08 | 未打开 near_obstacle | 通过 |
| C09 | seed/program 与 pilot/baseline/calibration 零重叠 | 五对交集全 0 |
| C10 | 检验锁 margin 绑定 | state=frozen、α/δ 逐位、散列一致 |

**分母**：清单 400 / 可用 400 / completed **377** / business_failure **23** / missing 0 / excluded **0**。

**C06 三格缺口**

| 格 | 完成/配额 | 完整率 |
|---|---|---|
| `arc_combined/combined` | 13/20 | **0.650** |
| `arc_combined/mass_only` | 15/20 | **0.750** |
| `start_stop_turn/mass_only` | 17/20 | **0.850** |
| `arc_combined/friction_only` | 18/20 | 0.900（贴线） |
| `straight_shift/combined` | 18/20 | 0.900（贴线） |

23 条业务失败的分布：按 profile 为 `combined` 10（arc_combined 7 / straight_shift 2 / alternating 1）、
`mass_only` 10（arc_combined 5 / start_stop_turn 3 / alternating 1 / straight_shift 1）、`friction_only` 2、
`clean` 1、`push_only` **0**；按族为 `arc_combined` 14 / `start_stop_turn` 3 / `straight_shift` 3 / `alternating` 3。
这与 §33 的 δ 构成（combined 12.50、mass_only 12.13 最高，push_only 12.39 但样本全过）方向一致：
**大载荷叠加扰动是真实弱项，不是抽样偶然**。对照校准数据的 2 格缺口（0.825 / 0.875），
heldout 有 3 格且最低到 0.650，说明该缺口在独立数据上被复现并放大。

**裁定**（用户 2026-11-07，沿用 HIRA-030 先例）：不补采、不改 20 条/格配额、不删不替换失败记录；
三格如实入档并**保留在主分母**，但不进入**格级**正式结论。HIRA-033 必须并列两套分母：
全体 400 条与剔除这三格的 340 条，不得只报一套。

**边界（下游必须知道）**

1. **格级结论只覆盖 17/20 格**。总体覆盖率与 profile/族/方向子组仍可报，但任何"逐格 90%"的说法
   必须标注这 3 格资格不足，且不得把它们剔除后再宣称"20 格全达标"。
2. **warmup 瑕疵本批未复现**：最短截断 364 步（1.82 s，`straight_shift/combined-0000`），
   23 条失败全部发生在窗口内，**0 条"开局即倒"**（对照校准 2/800 的 `non_computable`）。
   因此 §32 的 warmup 顾虑在 heldout 上没有新增证据，但仍不作为"下一版不必修"的理由。
3. **`check_heldout.py` 有一处显示瑕疵**：汇总行打印 `completed=377/800` 且前缀写作 `CAL_INTEGRITY`
   （从 `check_calibration.py` 抄来的硬编码），JSON 内的 counts 与 cells 是正确的。该脚本不在任何锁的
   `code.files` 内（锁里 `tests/` 只有 `run_split.py`），但按"已产出报告不重跑覆盖"的规矩本轮不改，
   留到 §43（HIRA-038/039）的已知瑕疵清单。
4. 本节**不打开 near_obstacle**，也不回写 α/δ 或功率模型（C08/C10 已机器化验证这一点）。


## §35 HIRA-033：独立检验的覆盖率、区间与适用范围（成本 0.9098 / 位置 0.8966 / 联合 0.8170）

**这一节解决什么**：§34 只回答"数据能不能用"，本节回答"冻结的 α/δ 在新条件下还管不管用"。
只用 `heldout` 的 400 条，α/δ、功率模型、清单一律从冻结产物加载，**不重算、不调整、不剔除**。

**工具**：`tests/coverage_heldout.py`（新增）。统计零件全部复用 `src/camra_calibration/coverage.py`，
本脚本只做 run 目录 → `HeldoutRow` 的适配。ε_d 与 HIRA-031 同口径：由 `(family, seed)` 用
`build_program` 重建命令、planar `integrate` 生成名义轨迹，取 10 Hz 配对网格上执行与名义的
**最大平面欧氏距离**；`ε_c = (J_exec − J_nom)/J_nom`，`J_nom < 1 J` 记 non_computable。

**两套分母（协议要求并列，不得只报一套）**

| 口径 | 期望 | 完整且可算 | 业务失败 | 缺失 | 数据排除 | 完整率 |
|---|---|---|---|---|---|---|
| 全体 400 条 | 400 | 377 | 23 | 0 | 0 | 0.9425 |
| 剔除 3 个资格缺口格 | 340 | 332 | 8 | 0 | 0 | 0.9765 |

**总体覆盖**（条件分母；区间为 cell 整簇 bootstrap 95%，10000×）

| 口径 | 成本 | 位置 | 联合 |
|---|---|---|---|
| 全体 400 条 | **0.9098** (343/377) [0.846, 0.962] | **0.8966** (338/377) [0.821, 0.959] | **0.8170** (308/377) [0.725, 0.897] |
| 剔除 3 格（340 条） | 0.9217 (306/332) [0.852, 0.976] | 0.9157 (304/332) [0.843, 0.976] | 0.8464 (281/332) [0.755, 0.929] |

**独立性诊断**：联合 / (成本 × 位置) = **1.0016**。两项误差近似独立，所以联合 81.70% 不是新的
结构性缺陷，而是两项边际各自不到 100% 的必然乘积效应——这条与 v2 的 CAL-027（比值 0.9972）同向。
它**不**构成"联合达标"的借口：联合覆盖仍是实测值，协议明令不得用边际乘积顶替。

**20 格分布**：17 格可用（完整率 ≥ 0.9），clean 档 4 格全部 1.0000。最弱的三格（按 §34 裁定不作结论）
为 `arc_combined/combined` 联合 0.3077、`start_stop_turn/mass_only` 0.7059、`arc_combined/mass_only` 0.7333。
另有两格虽完整率达标但值很低，需在 037 的比较里点名：`arc_combined/push_only` 位置覆盖 **0.5000**、
`start_stop_turn/combined` 成本覆盖 **0.5500**。

**按 profile**（全体口径）：clean 1.0000 / friction_only 0.9231 / mass_only 0.6714 / combined 0.6143（联合）。
扰动档是主要失分处，与 §33 的 δ 构成（combined 12.50、mass_only 12.13）完全一致。

**边界（下游必须知道）**

1. **位置覆盖 89.66% 比 90% 目标低 0.34 pp**。这是实测，不许通过调 α/δ、换分位或重挑 heldout 去补
   （协议 `statistical_acceptance.heldout.may_not`）。对外表述必须写成"未完全达到 90% 参照"，
   不能写成"达到 90%"。
2. **成本覆盖 90.98% 达标**，但与 v2 的 88.83% 不可直接比：v2 是补采后口径（`__a2` 重试），
   HiRA 是零重采口径。037 必须带上这条注脚。
3. **格级结论只有 17/20 格**。任何"20 格全部达标"的说法都是错的；报告已按 `--adjudicated-gap-cell`
   硬门禁，声明与实际不达标格不一致会直接 REFUSED。
4. **区间只反映抽样不确定性**：单一固定控制器、单一冻结裕度，不声称训练随机种子方差；
   重采样单位是整条轨迹 / 整簇，绝不按 50 Hz 时间点伪增 n。
5. 未覆盖 69 条（成本或位置任一不达标），明细在 `heldout_coverage.json#uncovered.rows`（前 50 条）。


## §36 HIRA-034：200 条近障检查与障碍注入补实现（188/200，两格资格缺口已裁定）

**这一节解决什么**：在冻结的 200 张近障地图上采 200 条，检查"障碍是否真的进了场景、位置是否对"，
为 §37（HIRA-035）的几何适用性提供可用数据。**本节不评估覆盖率与碰撞率**，那是 §37。

### 36.1 开跑前发现的一个硬缺口（文档 §9 是错的）

交接文档 §9 写「HIRA-034 现在可以直接用的东西：`run_split.py --split near_obstacle --maps ...`」，
只缺「先跑 check_margin_lock_binding.py」。**实际是缺了整个障碍注入实现**：

- `run_split.py` 的 `--maps` 只喂给 `enforce_collection` 做散列校验（第 140 行），之后没有任何消费者；
- `HiraCollector.__init__` 没有 maps 参数；
- `collector.py` / `env_runner.py` 里 `obstacle / maps / kinematic / slot` 关键字 **0 命中**；
- 场景（`env_runner.py` 的 `_SceneCfg`）里只有一个机器人。

照原样跑会得到 200 条"空地无障碍跑 30 秒"，而 §34 的验收第一条就是"检查障碍实际注入且逐 env 位置正确"。
发现后停下报告，用户裁定「补上障碍注入再采」。

### 36.2 补了什么

| 文件 | 改动 |
|---|---|
| `adapters/hira_go2/obstacles.py`（新增 104 行） | 7 个 kinematic 槽位、场地系→世界系位姿、逐 env 批量位姿与读回 |
| `adapters/hira_go2/env_runner.py` | `with_obstacles` 开关（**默认 False**）、有障碍时 `_SceneCfg` 显式声明 `obs_0..obs_6`、`set_obstacles` / `read_obstacles` |
| `adapters/hira_go2/collector.py` | `maps_path` 参数、按 trial_id 匹配地图 case、逐 env 写各自 `start_pose`、warmup 前设障碍、warmup 后读回、meta 落 `obstacles` 证据段 |
| `tests/run_split.py` | `--maps` 传给 collector、用地图 `start_pose` 覆盖 `sample_spec` 的随机初态、地图散列写进 `collect_config` |

三条设计约束（都写进代码注释）：

1. **障碍默认不进场景**。kinematic 刚体会改变 broadphase 与求解器状态，已采的 800 条校准 + 400 条检验
   必须在"无额外几何"下可复现，所以 `with_obstacles` 缺省 False，只有 near_obstacle 打开。
2. **每例起点不同**（地图 `start_pose`）。原先 `execute_batch` 是"全批共用 `specs[0].initial_pose`"，
   近障必须逐 env 写；`clean/mass_only/...` 仍走原路径，不给历史数据引入差异。
3. **绝不给障碍加自定义材质**（v2 CAL-028 的教训）：每个 `PreviewSurfaceCfg` 触发一次着色器编译，
   本机 ~85 s/槽位，会把场景创建从 4 s 拖到 >10 min。

### 36.3 检验锁 v1 → v2（红线 5 的直接后果）

`collector.py`、`env_runner.py`、`run_split.py` 三个都在 37 个 `code.files` 里，改一个字节旧锁即失效。
按「锁只能新增」新建 `data_lock_hira_evaluation_v2.json`（sha **`db156f4d`**）：

- 只重算 `code`（37 → 38，`changed = [run_split.py, collector.py, env_runner.py, obstacles.py]`）；
- controller / protocol / power_model / footprint / step_cost / asset / trial_lists / maps / margins / gates **逐字沿用 v1**；
- `stage` 仍写 `evaluation`（多个脚本硬校验该字符串），版本靠文件名 + 新增的 `amendment` 段区分；
- **heldout 400 条的归属仍留 v1**（`289e4137`），v1 与采集锁 v2 都未编辑、未覆盖。

门禁：补偿闸门 **10/10**、数据锁负测试 **13/13**（L12 确认地图仍是 v2 原件 `78c0145f`）。

### 36.4 B09 为什么必须改（控制变量，不是放宽）

`check_margin_lock_binding.py` 的 B09 原本直接拿采集锁测「margins 未校验」这个缺口。
但本轮改了代码，采集锁 v2 会**先在 code 层被拒**——缺口不是消失，是被更强的原因掩盖，
这个诊断项也就测不到它本来要测的东西。

改法：**只隔离 code 一个变量**——复制采集锁，把 `code` 段对齐当前代码树，其余（`margins=not_yet_produced` 等）
逐字不动，再用它测 margins 层。修后 B09 复现成功，detail 里同时记录「直测原始采集锁会被 code 层拒」。
这不是放宽判据：判据仍是"能否放行"，只是把不可比的干扰变量拿掉了。

### 36.5 采集结果

run `near_obstacle_20261107`：25 批、wall **442.9 s**、**225.2 MB**、`SPLIT_NEAR_OBSTACLE_RUN_OK` + `rc=0`，
200 个 trial 全部 `attempt_01`、零重采、单条 2.16–2.25 s（GPU0 空卡）。

| 项 | 结果 |
|---|---|
| 配额 | 200 条 / 20 格 / 每格 10 ✅ |
| **N03 障碍注入** | **200/200 非空；`n_obstacles` 与地图逐条一致（7/6/5）；累计 1338 个；位姿读回最大偏差 4.76e-07 m；起点逐条等于 `start_pose`** |
| 能量 | 200 条离线复算最差 rel **0.00e+00** |
| 计数 | completed **188** / business_failure **12** / missing 0 / excluded 0 / non_computable 0 |
| 结构 / 旋钮 / 身份 / 地图 / margin 绑定 | 全部通过 |

### 36.6 N07 两格缺口与裁定

| 格 | 完成/配额 | 完整率 |
|---|---|---|
| `arc_combined/combined` | 6/10 | **0.600** |
| `start_stop_turn/mass_only` | 6/10 | **0.600** |
| `arc_combined/mass_only` / `start_stop_turn/combined` / `straight_shift/combined` / `straight_shift/mass_only` | 9/10 | 0.900（贴线，零余量） |
| 其余 14 格 | 10/10 | 1.000 |

12 条失败**全部**在 combined(6) 与 mass_only(6)，clean 与 push_only 零失败。与 §32（0.825/0.875）、
§34（0.650/0.750/0.850）同型：**同一个"负重 + 扰动"弱项，场景越难越放大**。

**裁定**（用户 2026-11-07，第三次沿用同一先例）：不补采、不改 10 条/格配额、不删失败；
两格保留在主分母但不进格级正式结论。§37 必须并列全体 200 与剔除这两格的 180 两套分母。

### 36.7 边界（下游必须知道）

1. **`near_smoke_0919` 是冒烟 run（40 条），不计入 1820 配额**，保留作障碍注入的首轮验证证据。
   正式数据只有 `near_obstacle_20261107`。
2. **N04 首版误报 2 条推扰次数少 1 次**：期望时刻在 29.996/29.998 s，而推扰按 0.02 s 策略步调度，
   最后一个策略步是 29.98 s——物理上调度不到。已把判据改为「可调度时刻逐一对齐 + 终止 trial 取前缀」，
   首版报告留档为 `near_obstacle_collection_integrity_run1.json`。**这不是采集缺陷**。
3. **场地边界墙没有建模**（与 v2 一致，协议要求）：越界一律用冻结足迹做几何检查，不是物理碰撞。
4. **障碍是 kinematic 刚体**：它能拦住机器人，但不受力、不会移动；碰撞是"机器人撞静止柱"，
   不包含柱子被推倒的情形。
5. 本节**不评估**覆盖率、碰撞率与 δ 适用性；δ = 9.03 m 的几何预判（自由面积中位 0.0%）已在
   `reports/delta_feasibility.json`，§37 才会用这批数据给出实测结论。


## §37 HIRA-035：近障场景的几何适用性（δ 条件 0/200）

**这一节解决什么**：§36 只回答"这 200 条数据是不是真有障碍"，本节回答"在这套地图尺度下，
冻结的 δ 还有没有几何意义"。答案是没有，而且这个"没有"是可以用两种口径各自独立证明的。

### 37.1 三层证据，不许混为一谈

| 层 | 用什么算 | 回答什么 |
|---|---|---|
| 实测几何 | 执行轨迹（50 Hz）× 冻结足迹 × 冻结障碍 | 机器人**实际**离障碍多近、撞没撞、出没出场 |
| 名义几何 | 名义轨迹（10 Hz，`build_program` 重建 + planar 积分）× 同一足迹与障碍 | **计划路线本身**离障碍多近（与跟踪误差无关） |
| δ 条件 | 名义净距 vs δ | 危险集膨胀 δ 之后，这条路还安不安全 |

### 37.2 δ 条件：0/200，两种口径都是 0

| 口径 | 判据 | 满足条数 |
|---|---|---|
| 保守（足迹） | 足迹多边形到障碍的最小净距 ≥ δ | **0 / 200** |
| 最宽松（中心线） | 中心线到障碍 ≥ δ（直接用冻结地图自带的 `clearance_achieved_m`） | **0 / 200** |

名义最小净空中位 **0.053 m**、最大 0.565 m；地图自带中心线净距中位约 0.10 m、最大 **0.370 m**，
比 δ = 9.032 m 小 **8.66 m**。也就是说：**不是判据选得太严，是地图尺度与 δ 差了一个数量级**。

按协议 `statistical_acceptance.near_obstacle`：这种情况只作"记录与询问"，
**不得改分位缩小 δ、不得按新 δ 重设计地图**。用户已在 §33 阶段决定"照采、如实报告"。

### 37.3 实测几何

| 净距档位 | 条数 | 几何重叠 | 最小净空 | 中位净空 | 越界 |
|---|---|---|---|---|---|
| 0.05 m | 50 | 43 | 0.0000 | 0.0000 | 7 |
| 0.10 m | 50 | 46 | 0.0000 | 0.0000 | 3 |
| 0.20 m | 50 | 35 | 0.0000 | 0.0000 | 3 |
| 0.35 m | 50 | 27 | 0.0000 | 0.0000 | 2 |
| **合计** | **200** | **151（75.5%）** | **0.0000** | — | **15** |

档位越宽重叠越少（43 → 46 → 35 → 27），方向正确，说明几何计算没有退化成常数。

### 37.4 覆盖（两套分母并列）

| 口径 | 成本 | 位置 | 联合 |
|---|---|---|---|
| 全体 200（条件分母 188） | 0.9362 (176/188) | 0.9840 (185/188) | 0.9202 (173/188) |
| 剔除两格（180，条件分母 176） | 0.9489 (167/176) | 0.9886 (174/176) | 0.9375 (165/176) |

按 profile（联合）：clean 1.0000 / push_only 0.9750 / friction_only 0.9000 /
combined 0.8529 / mass_only 0.8529。

### 37.5 最关键的一条：δ 大 ≠ 安全

**位置覆盖 98.40%，几何重叠 75.5%。** 两个数同时成立，看起来矛盾，其实正是本节的结论：

- `ε_d ≤ 9.03 m` 几乎对每条 trial 都成立 → 位置覆盖很高；
- 但机器人照样在 3/4 的 trial 里和柱子几何重叠。

原因：**δ 描述的是"执行相对名义的跟踪误差上界"，不是"离障碍的碰撞裕度"**。地图本身是
贴着名义路径设计的（名义净空中位 0.053 m），机器人只要不精确沿名义路径走就会碰上；
而 9 m 的跟踪误差容忍度完全不能阻止这件事。

所以任何"位置覆盖 98% 所以近障安全"的读法都是错的。协议也明令：δ 是危险集扩张的
**必要条件而非充分条件**，可用性取决于尚未实现的上层导航策略（与 v2 的 CAL-029 同一措辞）。

### 37.6 与 v2 的量级对照

| 项 | v2（CAL-029，n=197） | HiRA（本节，n=200） |
|---|---|---|
| 几何重叠 | 138（70.1%） | **151（75.5%）** |
| 越界 | 13 | **15** |

同一量级。但分母口径不同（v2 的 200 个目录里含 2 个 `__a2` 补采），只作量级参照，
正式比较在 §39（HIRA-037）。

### 37.7 边界（下游必须知道）

1. **本节不是避障成功率验收**。控制器只是按冻结命令回放，没有上层导航在避障；
   把它读成"HiRA 避障成功率 24.5%"是错的。
2. **物理接触没有记录接触力**。`n_overlap=151` 是"足迹多边形与障碍多边形几何相交"，
   是碰撞的下界证据；真正的物理接触数与冲量本轮没有采集。
3. **场地边界墙未建模**（协议要求）：15 条越界是几何余量判定，不是撞墙。
4. **名义净空用的是 10 Hz 名义路径**（与 ε_d 同一网格），不是 200 Hz 积分；
   两者在 30 s 内的差异远小于本节讨论的量级（§33 已核：δ 的 50 Hz 与 10 Hz 差 0.0059 m）。
5. 格级结论只覆盖 18/20 格（§36 裁定的两格只报诊断值）。


## §38 HIRA-036：七分片独立审计（12/12，两条差异已解释）

**这一节解决什么**：前面 35 项都是"自己量自己"——用同一套函数算出来、再用同一套函数检查。
本节换一把尺子：**审计脚本不 import `camra_calibration` 与 `hira_go2` 的任何函数**，
把平面名义积分（圆弧闭式解）、二次功率特征、能量正功、nearest-rank 分位、覆盖判定、
矩形-足迹相交全部按规格另写一遍，只读它们的**产物**。

### 38.1 结果

| 编号 | 查什么 | 结果 |
|---|---|---|
| A01 | 七分片合计 1820、各片等于预登记配额 | 20/240/80/80/800/400/200 ✅ |
| A02 | `trial_id` 全局唯一 | 1820 → 去重 1820 ✅ |
| A03 | 每个 trial 有有效 attempt、主 attempt 唯一 | **1820 条全部只有 1 个 attempt（零重采、零 resume）** |
| A04 | 七个 seed 块两两不相交 | 21 对全空 ✅ |
| A05 | bundle / margins / model 与检验锁逐项一致 | `3c32fe31` / `57d05222` / `27bc5fe1` ✅ |
| **A06** | 能量从原始 200 Hz 重积分（1820 条） | 最差 rel **4.97e-16**（float64 舍入级） |
| **A07** | α/δ 从 771 条完整校准独立复算 | α / δ 与冻结值 **Δ = 0.00e+00（逐位一致）** |
| **A08** | heldout 与 near 六项覆盖率独立复算 | 最大偏差 **0.00e+00** |
| A09 | 近障几何重叠独立复算 | 148/200 vs 主报告 151/200（差 −3，见 38.2） |
| A10 | α/δ 只由 calibration 得出 | `split=calibration`、`n=771`、未引 heldout/near ✅ |
| A11 | 失败样本留在主分母 | 七片 `completed+bf+missing=manifest` 全部自洽 ✅ |
| A12 | v2 冻结物未被改动 | 侧车自校验 + 锁→margins 链一致 ✅ |

**A06/A07/A08 三条是本节的核心**：能量差 4.97e-16、α/δ 逐位相同、覆盖率零偏差 ——
说明前面所有数值不是"同一函数自证"，而是换实现后仍能复现。

### 38.2 两条差异（都已解释，无阻塞）

1. **A09 几何重叠 148 vs 151（差 −3）**：独立实现按 **10 Hz 抽样**、用「足迹顶点在矩形内
   或 矩形中心在足迹内」判据；主报告用 **50 Hz 全帧多边形距离**。独立侧是保守下界，
   方向正确（只会少报不会多报）。属抽样与判据差异，不是数据问题。
2. **A12 侧车口径不统一**：v2 的 `data_lock_v1.json` 侧车记**文件字节散列**，
   `margins_v1.json` / `power_model_v1.json` 侧车记**内容自散列**（等于文件内 `sha256` 字段）。
   首版判据只认字节散列，把三个**没有被动过**的文件误判成篡改——这是审计脚本自己的 bug，
   已改为两者任一成立即通过。核对结论：三者内容均未变，锁→margins 链一致。

### 38.3 边界（下游必须知道）

1. **独立性有明确范围**：独立的是"从产物到数值"这一段；`tau`/`qdot` 本身来自采集器的记录，
   若采集器写盘时就算错，本审计发现不了（那由 §21 的 2000 子步逐位对照负责）。
2. **几何是保守下界**，不是逐条等价；要逐条一致得用同一套多边形距离库，那就失去独立性了。
3. **v2 只有"未改动"证据，没有"可复算"证据**：v2 的训练期原始日志（TASK-024/026/036）
   不在本机，§39 与 v2 对比时只能引用报告级数字，这条限制必须写在对比表脚注里。
4. 审计只读，未修改任何 trial 记录、未重采、未调整任何冻结值。


## §39 HIRA-037：与旧 v2 的报告级对比（未复算、非配对）

### 39.1 先说证据等级

v2 的逐 trial 原始数据（68 GB trials）**已不在本机**，四处核实：

| 位置 | 结果 |
|---|---|
| `/pub/data/caohy/logs/camra` | 3.1 GB，只有 `TASK-*` 日志，**无 `trials/`** |
| `/pub/data/caohy/camra_run` | 不存在（§12.5 提到的只读快照已不在） |
| `/home/caohy/logs`（NFS） | 233 MB |
| `Paper/.local` 备份 | 不存在 |

所以本节的 v2 数字**全部来自其冻结产物与交接笔记**（逐行标注出处），
`recomputed=false`、`evidence_level=report_level_only`。按计划 `constraints_and_notes`，
此情形必须先询问；用户已同意接受这一有边界的对比。

**做不了的事**：成对差异、簇级区间、"同 seed 同命令"的配对检验。清单层面
baseline/heldout/near 与 v2 已有 100% 模板匹配（§26），但缺 v2 的逐 trial 轨迹与扰动 trace。

### 39.2 配置可比性（10 项，4 项不可比）

| 维度 | 判定 | 关键差异 |
|---|---|---|
| 命令域 / 生成器 / 五 profile / 时序 | **可比** | 逐字段继承协议，同一 `build_program`，时序逐项实测 |
| 能量口径 | 可比（须注明） | 同公式；v2 float32 乘法、HiRA float64，系统差 ~2.6e-10 相对 |
| 协议版本 | 可比（须注明） | v1.0.2 → HiRA v1，差异 9 项全为身份/簿记 |
| **机器人资产** | **不可比** | unitree_ros USD vs legged_gym URDF（29 links / 27 colliders） |
| **执行器 PD** | **不可比** | **kp 25.0 → 20.0**（kd、action_scale 相同） |
| **力矩限幅** | **不可比** | 各自来源的 URDF effort |
| **重试口径** | **不可比** | v2 有 D-007（+31）/D-013（+2）整批重采；HiRA 1820 条全部单 attempt |

### 39.3 指标对比

| 指标 | v2 | HiRA | 可比性 |
|---|---|---|---|
| α | 0.1905（790 completed） | **0.3516**（771） | 同口径，1.85× |
| δ | 2.6139 m | **9.0322 m** | 同口径，3.46× |
| heldout 成本 | 0.8690（345/397） | **0.9098**（343/377） | 分母不同 |
| heldout 位置 | 0.9144（363/397） | **0.8966**（338/377） | 分母不同 |
| heldout 联合 | 0.7935（315/397） | **0.8170**（308/377） | 分母不同 |
| near 成本 | 0.8883（175/197） | **0.9362**（176/188） | 分母不同 |
| near 位置 | 0.9340（184/197） | **0.9840**（185/188） | 分母不同 |
| near 联合 | 0.8274（163/197） | **0.9202**（173/188） | 分母不同 |
| near 几何重叠 | 138/197 = 70.1% | **151/200 = 75.5%** | 同判据、分母不同 |
| near 越界 | 13 | **15** | 同判据、分母不同 |
| 功率 validate q90 | 0.0554（n=79） | 0.0588（n=80） | 同口径 |
| 足迹面积 | 0.4580 m²（43 顶点） | 0.6769 m²（52 顶点） | 资产不同 |
| 校准完整率 | 790/800 | 771/800 | 同口径 |
| 格级缺口 | 20/20（靠补采） | 17/20（不补采） | **重试政策不同** |

### 39.4 三条不能过度解读的地方

1. **方向不一致 ≠ 更好或更差**：heldout 成本 +4.08 pp、位置 −1.78 pp。一个变好一个变差，
   任何"HiRA 优于/劣于 v2"的单句结论都是错的。
2. **near 覆盖率被更大的 δ 抬高**：位置 0.9340 → 0.9840 看着漂亮，但 §37 已证明
   在 98.40% 的位置覆盖下仍有 **75.5% 的几何重叠**。δ 从 2.61 涨到 9.03，覆盖变高是算术必然。
3. **格级缺口的差异是政策差异**：v2 靠 D-013 一次性补采把 8/10 抬到 9/10，才达到 20/20；
   HiRA 按新契约不补采。这是**重试政策**差异，不是能力差异，不能列进"v2 更强"的证据。

### 39.5 一处更正（HIRA-033 的引用错误）

§35（HIRA-033）的 `remaining_work` 里写「v2 的 CAL-027（cost 0.8883 / position 0.9340 /
joint 0.8274，n=197）」——**这组数字是 v2 的近障**（§32.4，n=197 补采后），不是 heldout。
v2 的 CAL-027（heldout）实为 **0.8690 / 0.9144 / 0.7935（n=397）**。本节按后者引用。
计划里的历史 work_record 不修改（append-only），更正记在本节与 HIRA-037 的记录里。

### 39.6 边界

1. 本报告**不是**"HiRA 更好/更差"的判决书，是**描述性对照表**，含 4 项显式不可比。
2. 控制器包同时换了权重、默认姿态、kp（25→20）与机器人资产——**任何指标差异都不能归因于网络权重一项**。
3. v2 的旧文件（交接笔记、计划、锁、报告）**一字未动**，本节只读引用。
4. 若日后找回 v2 的 trials，可升级为复算 + 配对；当前版本不得被引作"已复算"。


## §40 HIRA-038：可复现交付包与独立目录重载（20/20 实跑通过）

**这一节解决什么**：前面 37 节都在**当前目录**里验证。本节把三棵树搬到**全新根目录**，
只重定位锁内路径，然后在那里把"加载 → parity → 采集 → 误差 → 覆盖 → 离线复算"整条链重跑一遍。
证明的是"这个包离开当前机器布局还能用"，不是"再跑一遍得到同样数字"（后者由 §38 的独立审计负责）。

### 40.1 两个拦路问题（都不是配置问题）

**① 锁里全是 `/pub` 绝对路径**。`code.files` 的键、`controller.*`、`protocol`、`margins`、
`trial_lists`、`gates`、`near_obstacle_maps` 共 **57 处**是绝对路径，而 `lock._resolve()`
对绝对路径原样返回——换个目录，`verify_artifacts()` 查的还是**原位置**，
"独立重载"就变成了自己骗自己。

处置：新增 `tests/delivery_relock.py`，把这些前缀改写成相对路径（`CaMRA/...`），
**所有 sha256 逐字保留**，原锁一字不动（红线：锁只增不改），派生文件另加 `delivery_rebase` 段说明来路。
生成后立刻 `DataLock.load(verify_hash=True)` + `verify_artifacts(new_root)` 自证：**0 项不符**。

**② 输出目录没有绑定校验**。`run_split.py` 只在**首次**写 `collect_config.json`，启动时
**不校验**既有 run 目录绑的是哪个 split / 哪把锁 / 哪个 bundle。实测（`plan_attempts` 探针）：
把一个 `collect_config` 写着 `split=heldout` 的目录当 `--runs-root` 用，它照样给出 `run_initial`，
会把 near 的数据写进去，一声不响。

处置：`run_split.py` 与 `collector.py` 都在锁内，**没有擅自改**。在交付层新增
`tests/run_delivery.py` 作为唯一正式入口：先核验 split / 锁散列 / bundle_id / 清单散列，
不符即 **exit 2 拒绝**；一致才 exec `run_split.py`。`--preflight-only` 可只核验。
**残留**：这道守卫在交付层，不在 `run_split` 里面；要下沉需新版锁（v3），列为待裁定项。

### 40.2 新增的四个脚本

| 脚本 | 职责 |
|---|---|
| `tests/build_delivery_manifest.py` | 按 17 个角色分区索引 220 个随包文件 + 9 个 run / 1880 个 trial 的大数据索引；`--verify` 可独立复核 |
| `tests/delivery_relock.py` | 锁内 57 处绝对路径改写成相对路径，生成派生锁（sha256 不动） |
| `tests/run_delivery.py` | 交付层采集入口：输出目录绑定守卫 + 原样 exec `run_split.py` |
| `tests/verify_delivery_reload.py` | 独立根下 D01–D20 二十项实跑验证 |

### 40.3 二十项验证（`reports/delivery_reload_verification.json`，20/20）

| 编号 | 查什么 | 结果 |
|---|---|---|
| D01/D02 | 工作根不在源仓库内、无指向源目录的软链 | 通过 |
| D03 | 三棵树逐文件散列与源一致 | 232 文件 / 9.3 MB，不一致 **0** |
| D04 | 副本脚本 `__file__` 自定位到副本 | 通过 |
| D05/D06 | 派生锁生成；新根下 `verify_artifacts` | 57 处重写，**0 项不符** |
| D07 | 源仓库检验锁 v2 未被改 | `db156f4d` 侧车一致 |
| D08 | **模型被换**（policy.pt 改一字节） | 拒绝：`721a891a` vs `d769fa8f` |
| D09 | **权重缺失** | 明确报「文件清单与目录不一致」 |
| D10 | **地图被换**（字节改动） | 门禁拒绝：`--maps` 散列与锁不符 |
| D11 | **资产缺失** | preflight 列出 `asset.geometry` 不存在 |
| D12 | **输出目录误换** | 交付入口 rc=**2** 拒绝；新目录 rc=0 放行 |
| D13 | bundle 重载 | `3c32fe31…`、policy `721a891a…`、arch `[512,256,128]` |
| D14 | obs/动作 parity（9 fixture + 5 类负测试） | `all_passed=true`，fixtures 散列与源报告一致 |
| D15 | 小批采集（走交付入口 + 派生锁） | 40 条、**104.1 s**、45.7 MB、`rc=0` |
| D16 | 官方 runs 目录 | **零新增、零 mtime 变动** |
| D17 | 障碍注入 | **40/40** 条 `meta.obstacles` 非空 |
| D18 | ε_c/ε_d 与覆盖判定（小批） | n=40，成本 38、位置 40 |
| D20 | 16 份核心报告可定位 | 缺失 0、散列不符 0 |
| D19 | 官方数据离线能量复算 | 12 条，最差 rel **2.704e-16** |

### 40.4 包索引（`delivery/bundle_manifest_hira_v1.json`）

- 17 个角色分区、**220 个文件**、8.0 MB、`missing=0`；构建时回读自校验 **220 / bad 0**；
  独立 `--verify` 复核 **files ok=220、raw_trials ok=1880、bad=0**。
- 大数据**不打包**：9 个 run / **1880 个 trial** / **2.13 GB**，按"逻辑相对路径 + trial 级树散列"
  索引，附取用说明（挂载路径或改 `--runs-root`）；USD 缓存 18.8 MB 只记摘要（可由 URDF 重建）。
- 明确**不打包**：稿件（红线）、凭证与 fakehome、已不在本机的 v2 逐 trial 数据。

### 40.5 边界（下游必须知道）

1. **D18 的覆盖率不是结论**：40 条小批只为证明链路能跑，正式口径仍以
   `reports/heldout_coverage.json`（377 条）与 `reports/near_obstacle_evaluation.json`（188 条）为准。
2. **`delivery_reload_20261107` 是验证 run，不计入 1820 配额**，与 `near_smoke_0919` 同性质。
3. **派生锁只能在新根用**：`data_lock_hira_evaluation_delivery.json` 的路径是相对的，
   拿回原布局会解析错。原锁 `db156f4d` 仍是 heldout 400 条的归属凭证。
4. **包索引不自我引用**：`delivery/` 本身不进 `artifacts`（否则散列自指）。
5. 本节的 20 项**只证明"包可用"**，不重复 §38 的数值独立复算，也不改变任何冻结值。


## §41 HIRA-039：开发入口文档同步与只读稿件差异清单

**这一节解决什么**：让"下一个接手的人"不用翻 40 节笔记就能分清 v2 与 HiRA、
找到模型/数据/证据和正确入口；并把稿件与实测的差异**列出来**（只列，不改）。

### 41.1 同步了哪 5 份文档

| 文件 | 改了什么 |
|---|---|
| `AGENTS.md` | 进度 0/40 → **38/40**（下一步 HIRA-039/040）；"尚未实施适配" → 已冻结的适配/锁/1820 条/α·δ/交付包；证据边界表里两处"尚未建立"改为"已建立"；补指向本报告与 `reports/manuscript_sync_differences.md` |
| `README.md`（根） | 进度与现状段改为实测结论（α/δ、三项覆盖、近障 0/200）；模块状态表两行更新 |
| `CaMRA/README.md` | 进度；两处入口"尚未建立"→ 已建立；**新增 v2 / HiRA 契约对照表**（观测缩放、关节顺序、增益、限幅、冻结裕度） |
| `CaMRA/calibration/README.md` | 进度；采集入口改为 `run_delivery.py`/`run_split.py`+锁（明确不能沿用 v2 命令）；**新增 HiRA 独立产物表**（11 个阶段 → 文件 → 结论） |
| `CaMRA/locomotion/README.md` | 进度；冻结包位置与 bundle/policy 散列；**新增 4 条 HiRA 与 v2 的契约差异**（不可互相套用） |

旧 v2 计划与旧结论**一字未改**（32/32、`overall_pass=False`、低速边界、§32 补采记录均保留）。

### 41.2 稿件差异清单（`reports/manuscript_sync_differences.md`）

只读对照 `manuscripts/CCDC/latex/1029/main.tex`（693 行）。分五组：

- **A 必须处理的数值差异（6 条）**：`main.tex:493` α=0.18 / δ=0.12 m → v2 冻结 0.19048 / **2.6139 m**、
  HiRA 0.351568 / **9.032231 m**（δ 差 20 倍以上；v2 笔记 §27.4 已注明"0.12 m 的旧值来自无依据的假设"）；
  `main.tex:490-492` 中位 0.09 / 0.06 m → 实测 ε_c 中位 **+0.0095**、ε_d 中位 **1.113 m**（都差一个量级，
  需作者先确认是否笔误或另一套归一化）；held-out 覆盖 353/400·346/400·336/400 → v2 345/397·363/397·315/397、
  HiRA 343/377·338/377·308/377；Fig.5 的 α/δ 标注；消融网格 `α∈{0,.09,.18,.27}`、`δ∈{0,.06,.12,.18}`
  在 δ 锚点为 2.61 m 时只覆盖 7%，属**结构性**问题，需作者决定重设还是注明前提。
- **B 口径表述（3 条）**：成本单位 a.u. → 实现为正关节机械功（J，`J_min=1 J`）；
  硬件 "50 Hz/5 Hz" → 实现 200/50/10 Hz（需作者确认哪套为准）；校准窗口 30 s 与硬件 60 s 需区分。
- **C 尚未包含的新结论（2 条）**：近障 δ 条件 0/200、位置覆盖 98.40% 与几何重叠 75.5% 并存
  → 建议在 limitation 补 2–3 句（负结果如实写更有说服力）。
- **D 已核对一致（4 条）**：800/400 规模、边际定义、ε_c/ε_d 定义、"不是导航成功率"限定。
- **E 复核方式**：`git status --short -- manuscripts/` 为空；本轮每次提交的暂存清单 grep 门禁均输出
  `OK no manuscript`。
- **F 需作者判断 3 件**：held-out 报哪一版分母、消融网格是否重做、50/5 Hz 与 50/10 Hz 哪个是硬件真实配置。

**全程未写入 `manuscripts/**`、任何 `*.tex`/`*.bib`、PDF，未重新编译**（`main.tex` 散列 `83fdb330…`、
`references.bib` `1991c0af…`，与本轮开始时一致）。

### 41.3 目前仍未解决的两件事（带到 §42）

1. **输出目录守卫在交付层**（`tests/run_delivery.py`），原生 `run_split.py` 仍不校验既有 `collect_config`；
   下沉需新版锁 v3，属待裁定（§40.1）。
2. **近障 δ 条件 0/200 是负结果**，不是待修缺陷：δ=9.03 m 与 8×10 m 场地差一个量级，协议禁止改分位或重设计地图。


## §42 HIRA-040：逐任务对账与最终交接（40/40，六类必需检查全过）

**这一节是交付的收口**：把"验收"写成可重复执行的脚本 `tests/final_acceptance_hira.py`，
六类检查各自输出布尔值与明细，任一类不符即返回码 1。产物 `reports/final_acceptance.json`。

### 42.1 对账结果

| 检查 | 判据 | 结果 |
|---|---|---|
| 计划结构 | 40 项、id 唯一有序、依赖指向更早任务、完成项必须有 verification_result + work_record + evidence_outputs | 0 问题 |
| 逐项证据 | `evidence_outputs` 里声明的路径逐条存在（多目录回退） | 缺失 **0** |
| 门禁机读判定 | 取报告里的判定字段，不是"文件存在" | **19/19** |
| 采集台账 | 官方七分片 1820/1820、每 trial 恰一个 attempt | **1820/1820，多 attempt 0** |
| 无覆写 / 无隐性重训 | 稿件、v2 四件冻结物、v2 计划、HiRA 子模块 | 全部未变 |
| 交付索引重算 | `build_delivery_manifest.py --verify` 重算 221 文件 + 1880 trial | **bad=0** |

`verdict.all_required_passed = true`，`failed = []`。

**门禁明细（19 项）**：观测 13、动作/PD 10、三时钟 11、能量计量 17、G1 16、采集链路 21、
pilot 10、基线 400 条、功率模型、足迹、步成本、α/δ 冻结（10+10）、独立审计 12、交付重载 20
——判定字段全为 true。三处**已裁定例外**按其预期值核对，而不是网开一面：

- HIRA-030：`verdict.failed = ["C06-cell-full-rate"]`（校准 0.825 / 0.875）；
- HIRA-032：`verdict.failed = ["C06-cell-full-rate"]`（heldout 0.650 / 0.750 / 0.850）；
- HIRA-034：`verdict.failed = ["N07-cell-full-rate"]`（近障两格 0.600）；
- HIRA-033：`full_sample.verdict.status = "insufficient_cells"`。

### 42.2 交付了什么

**新增代码**（本轮全部，均在仓库内）：适配层 13 个模块（含 `obstacles.py`）、校准库 24 个、
实验脚本 46 个（其中本轮新增 8 个：`coverage_heldout` / `evaluate_near_obstacle` /
`full_independent_audit` / `compare_hira_v2` / `build_delivery_manifest` / `delivery_relock` /
`run_delivery` / `verify_delivery_reload` / `final_acceptance_hira`）。

**真实采集数量**：官方 1820 条（20 pilot / 400 功率基线 / 800 校准 / 400 检验 / 200 近障），
全部单 attempt、零重采、失败样本零删除。非配额另有两批：`pilot_20261105T223428`（20）、
`near_smoke_0919`（40），以及交付验证 run `delivery_reload_20261107`（40，在独立工作根）。

**工程门禁**：G1 16/16；数值 parity 1.79e-7（f32 obs）/ 1.9e-6（τ）；三时钟 6000/1500/300/301；
能量 17/17；采集链路 21/21；数据锁 v1→v2 各 13/13；补偿闸门 10/10。

**科学结论**（唯一权威副本 `config/margins_hira_v1.json`）：α = **0.35156762163783695**、
δ = **9.032231244495946 m**（n=771，nearest-rank + 簇 bootstrap）。独立检验条件分母 377 条：
成本 **0.9098**、位置 **0.8966**、联合 **0.8170**；联合/(成本×位置) = 1.0016（两项独立，
联合缺口是边际的必然）。近障：δ 条件 **0/200**、几何重叠 **151/200**、越界 15。

### 42.3 能力限制（写进交付说明，不许淡化）

1. **工程接入通过 ≠ 随机安全理论证明，也不构成导航性能证明**。本轮没有任何导航训练/评估。
2. **位置覆盖 0.8966 低于 90% 参照 0.34 pp**；联合 0.8170 低于 90%。均为实测，未回填 α/δ。
3. **δ 与近障地图量级不匹配**：δ=9.03 m 对 8×10 m 场地，自由面积中位 0.0%、起点终点 0/40 自由。
   结论是"该场地尺度下近障不可评估"，不是"避障成功率 0"。
4. **与 v2 只能报告级对比**（v2 逐 trial 数据已不在本机，`recomputed=false`）。控制器包同时换了
   权重、默认姿态、kp（25→20）与资产，任何单指标差异不能归因于网络权重一项。
5. **格级结论只有 17/20（heldout）与 18/20（近障）**；三格/两格完整率不足者保留在主分母但不入格级结论。
6. **warmup 有 1 s 零动作前置**（与 800 条校准同口径），本版不修。

### 42.4 后续最直接入口

| 目的 | 入口 |
|---|---|
| 上层导航消费 | α 扣预算、δ 膨胀危险集（`config/margins_hira_v1.json`）；近障任务在该场地不可评估 |
| 复现/换机 | `docs/reproduction.md` + `delivery/bundle_manifest_hira_v1.json`（221 文件 / 1880 trial 索引） |
| 重新验收 | `python tests/final_acceptance_hira.py`（六类检查，返回码 0/1） |
| 采集新数据 | `python tests/run_delivery.py --split … --lock …`（带输出目录绑定守卫） |
| 稿件修改 | `reports/manuscript_sync_differences.md`（6 条数值 + 3 条口径 + 2 条新增结论 + 3 件待判断） |

### 42.5 两处登记瑕疵（未改历史记录）

1. **计划的 evidence_outputs 有两处文件名与实际不符**：HIRA-022 写 `pilot_end_to_end.json`
   （实为 `pilot_validation.json`）、HIRA-024 少写 `reports/` 前缀。按"历史记录只追加"未改计划，
   在对账脚本里显式登记别名并在报告 `task_evidence.rows[].aliased` 列出。
2. **`check_heldout.py` 汇总行分母硬编码 `/800`、前缀写作 `CAL_INTEGRITY`**（从校准脚本抄来）。
   JSON 数据正确，脚本不在任何锁内，按"已出报告不重跑"保留。

### 42.6 唯一未决的工程选择

**输出目录守卫是否下沉到 `run_split.py`**：目前守卫在交付入口 `tests/run_delivery.py`，
因为 `run_split.py` 在锁的 `code.files` 内，下沉必须出新版锁 v3。三种处置（保持现状 / 出 v3 锁 /
交 040 后另行裁定）已报用户，未自行决定。


## §43 HIRA-040 后加固：把输出目录守卫下沉到 run_split（检验锁 v3）

**这一节解决什么**：§40.1 登记的那个缺口——`run_split.py` 启动时不校验既有 `collect_config`，
误换 `--runs-root` / `--run-id` 会把新数据静默写进别的 split 的目录——已按用户 2026-11-08 授权
**下沉到采集脚本本身**，不再是"只靠交付入口拦"。

### 43.1 改了什么（只改一个文件，不碰控制层与模型）

| 层 | 是否改动 | 说明 |
|---|---|---|
| 控制层（观测 / PD / 关节映射 / 策略加载） | **未改** | `observation.py`、`action_pd.py`、`bundle.py`、`env_runner.py` 一字未动 |
| 冻结模型（`policy.pt` + `controller.json`） | **未改** | 散列仍是 `721a891a…` / bundle `3c32fe31…` |
| 采集入口 `tests/run_split.py` | **改了** | 新增 `_check_run_dir_binding()`，写 `collect_config.json` 前比对四项 |

四项判据：目标 run 目录里已有 `collect_config.json` 时，比对

1. `split` 是否一致；
2. `data_lock.sha256` 是否一致；
3. `bundle_id` 是否一致；
4. `manifest_sha256` 是否一致。

任一不符即打印 `[guard] 拒绝启动：输出目录已绑定另一批配置 —— …` 并 **exit 2**，**一个字节都不写**。
目录不存在或没有旧卡（首次采集）照常放行。

### 43.2 检验锁 v2 → v3（红线 5 的直接后果）

`run_split.py` 在 v2 的 `code.files` 里，改一个字节 v2 即失效。按"锁只能新增"派生
`config/data_lock_hira_evaluation_v3.json`（sha **`d7d60cb9d28f844e`**）：

- 代码 38 → 38 个文件，**变化仅 `run_split.py`**；
- controller / protocol / power_model / footprint / step_cost / asset / trial_lists /
  near_obstacle_maps / margins / gates **逐字沿用 v2**；
- margins 仍是 `57d05222…`、`state=frozen`，α/δ 未动。

**归属不变**：heldout 400 条仍归 v1（`289e4137`）、near 200 条仍归 v2（`db156f4d`）。
v1/v2 一字未改，v3 只用于此后的新采集。

### 43.3 门禁与两次冒烟（都真跑）

| 项 | 命令 | 结果 |
|---|---|---|
| 数据锁负测试 | `check_data_lock_hira.py --lock …v3.json` | **13/13**（`reports/formal_gate_evaluation_v3_1900.json`） |
| 补偿闸门 | `check_margin_lock_binding.py --lock …v3.json` | **10/10**（`reports/margin_lock_binding_v3_1900.json`） |
| 负冒烟 | 指向一个已绑 heldout 的沙箱目录 | **exit 2 拒绝**，目录内只留原来那张卡，**0 条 trial 写入** |
| 正冒烟 | 全新目录采 40 条 near | `SPLIT_NEAR_OBSTACLE_RUN_OK`、121.4 s、44.7 MB、**40/40 条带障碍证据** |

负冒烟的实际输出（这就是这道守卫的直接证据）：

```text
[split] 0.9s [guard] 拒绝启动：输出目录已绑定另一批配置 ——
        split 目录='heldout' 本次='near_obstacle'；
        data_lock 目录=289e41379bd19cd4 本次=d7d60cb9d28f844e；
        manifest_sha256 目录记录与本次清单不符
EXIT=2
```

### 43.4 一个必须记下的伪失败

首次跑 13 项负测试时 **L13-resources 失败**（"并发采集进程 1"）。追查发现不是真有采集在跑：
`check_data_lock_hira.py` 用 `pgrep -f "run_[s]plit"` 判并发，而我有一条**卡在 NFS 上的探查命令**
（`git status` 挂住），它的命令行里正好包含 `run_split.py` 这个字符串，被算成了采集进程。

处置：kill 掉自己那条残留 PID（**只 kill 具体 PID，不用 pkill -f**），重跑即 13/13。
**这不是代码缺陷，是检查器的匹配口径偏宽**——任何命令行里出现 `run_split` 字样都会被算进去；
记在这里，避免下次误判成"有人偷偷在采"。

### 43.5 边界

1. 守卫在 **AppLauncher 已创建之后**才触发（实测 0.9 s 退出，未进入仿真主循环、未写数据）；
   要更早失败得把校验提到 launcher 之前，属可选优化，本轮未做。
2. 守卫只防"写错目录"，**不防"用错锁"**——锁与现场的散列校验仍由 `enforce_collection` 负责，两者互补。
3. `run_delivery.py` 保留：它在 exec `run_split.py` 之前还会校验一次，且是文档化的正式入口；
   现在等于**双层拦截**（入口一层、采集器一层）。
4. 交付验证报告 `reports/delivery_reload_verification.json` 里的 D12 仍以交付入口为准，
   其结论"误换目录被拒"在两层都成立；本次新增的负冒烟是**采集器层**的直接证据。


## §44 执行主机记录与迁移要求（2026-11-08）

**为什么单独记一节**：本项目前后跨了两天、跨了多个会话，加上 `/home` 是共享 NFS，
"到底在哪台机器上跑的"很容易被记混——打开仓库看到文件，不等于在跑那台机器上执行。
这一节把执行主机写成可核对的事实，并落实用户 2026-11-08 的迁移指令。

### 44.1 本任务（HIRA-001–040）全部执行在 `tiger12`

| 证据 | 内容 |
|---|---|
| 主机名 | `hostname` → `tiger12.ios.ac.cn` = `192.168.20.12`（交接文档 §2 第 39 行原定目标机） |
| 数据位置 | 1820 条官方采集 + 6 批非配额 run 全在 `/pub/data/caohy/logs/camra_hira/runs/`，共 9 个 run 目录 |
| 工作副本 | `/pub/data/caohy/work/hira/CaMRA`（锁内 `code.files` 的键全部指向这里） |
| 解释器 | `env_isaaclab` = Python **3.11.18**（`.11` 上是 3.11.15，可据此区分） |
| 盘的性质 | `/pub` 是**节点本地盘**：`.11` 的 `/pub/data/caohy` 只有 `IsaacLab` 与 `miniconda`，**没有 `work/`、没有 `logs/`** |

### 44.2 另外两台的角色（本次均未执行任务）

| 机器 | 主机名 | 本次用途 | 数据 |
|---|---|---|---|
| `192.168.20.11` | tiger11 | 2026-11-08 当**只读检查/跳板**；**2026-11-09 起为用户指定的后续执行机**（见 44.5） | **无** HiRA 数据（`/pub/data/caohy` 只有 `IsaacLab`、`miniconda`） |
| `192.168.20.10` | tiger10 | **未使用**；原列为备用登录入口，2026-11-09 用户明确不使用 | 未核验 |

### 44.3 2026-11-08 的登录故障（已定位，非机器宕机）

表现：TCP 能连上 22 端口，但**收不到 SSH banner**，客户端报
`Connection timed out during banner exchange` 或 `Connection closed`；
同期用户的 VS Code 仍能打开仓库目录（因为它连的是共享 NFS 上的另一台节点，看到的是同一份文件）。

定位：从 `.12` 反探 `.10` 的 22 端口，得到明文 `Exceeded MaxStartups` ——
即 **sshd 的"未认证连接槽位"被打满**。`.12` 的 `/etc/ssh/sshd_config` **没有显式配置** `MaxStartups`，
用默认 `10:30:100`；并发未认证连接超过 10 个时，新连接直接被丢弃，表现为"能连但不发 banner"。
槽位被不可中断的卡死子进程占住时，这个状态能持续数十分钟（本次约 50 分钟）。

**运维建议**（需系统管理员执行，本任务不碰 `/etc`）：把 `MaxStartups` 放宽到 `30:60:200`，
并定期清理 `STAT=D` 或超长 `etime` 的 `sshd` 子进程。

### 44.4 迁移指令：后续任务在 `192.168.20.11` 执行（用户 2026-11-08 指令，2026-11-09 更正目标机）

这条已写入 `AGENTS.md` 的「执行主机」一节，作为后续会话的硬要求。**目标机由 `.10` 更正为 `.11`**
（用户 2026-11-09；`.10` 不使用）。落地前必须确认三件事，否则会出现"换了机器但数据不在"的死局：

1. **`.11` 的 `/pub` 是否有同一套数据（实测：没有，必须先搬）**。当前 HiRA 的 1820 条、工作副本、USD 缓存
   都在 `.12` 的本地盘上，而 `.11` 的 `/pub/data/caohy` 只有 `IsaacLab` 与 `miniconda`；必须显式复制
   （`runs/` 约 2.1 GB + `work/hira/CaMRA` + `assets/usd_cache`），
   复制后按 `delivery/bundle_manifest_hira_v1.json` 的 trial 级散列逐项核对。
2. **锁内绝对路径是否仍然成立**。三把锁的 `code.files`、`controller`、`margins`、`trial_lists`、
   `gates`、`near_obstacle_maps` 全是 `/pub/data/caohy/work/hira/CaMRA/...` 绝对路径；
   只要 `.11` 上**布局完全相同**就无需改锁；若路径不同，用 `tests/delivery_relock.py` 生成派生锁
   （只改前缀、散列不动），**不得改写原锁**。
3. **`.11` 上的环境与设备**。`/pub/data/caohy/miniconda/envs/env_isaaclab` 在位、Python **3.11.15**
   （`.12` 为 3.11.18，故 `config/hira.env` 的根路径必须按 `.11` 重设）；机器有 GPU（`nvidia-smi -L`
   可见多张 RTX 4090），Isaac Lab 5.1/0.47 一套需按 `.11` 的实际路径确认后再跑采集。

**红线补充**：不得在 `.12` 与 `.11` 上**同时**采集同一 split——会产出两套 run 目录，
而"失败数据零删除"意味着两套台账都要留着，事后无法说明哪套是正式数据。
一次只允许一个执行机持有采集权，切换时在 `collect_config.json` 与计划记录里写清。

### 44.5 更正记录（2026-11-09）

用户 2026-11-09 明确：**后续任务使用 `192.168.20.11`，`192.168.20.10` 不使用**。
2026-11-08 写入的「必须在 `.10` 执行」有误，本日已在三处同步更正：
`AGENTS.md` 的「执行主机」一节、本节 44.2 与 44.4、`docs/reproduction.md` 的「前置」。

- **事实不变**：HIRA-001–040 仍全部执行在 `tiger12`；`.11` 当时只作只读检查/跳板。
- **前提不变且更明确**：`.11` 的 `/pub/data/caohy` 目前只有 `IsaacLab` 与 `miniconda`，
  `work/`、`logs/` 均不存在 —— 迁移前必须显式复制并逐 trial 核对散列（见 44.4 第 1 条）。
- **§44.3 的故障记录保持原样**：那次是直接探测 `.10` 的 22 端口得到 `Exceeded MaxStartups`，属当时事实；
  `.12` 同现象，两者都是 sshd 未认证槽位打满，与"用哪台机器执行"无关。


## §45 数据迁移：tiger12 → tiger11（2026-11-09，1880/1880 trial 核验通过）

**为什么搬**：§44.5 把执行主机改成了 `.11`，但 `.11` 的 `/pub` 是**空的**（只有 `IsaacLab` 与 `miniconda`）。
`/pub` 是节点本地盘、不共享，所以 1820 条采集与工作副本必须显式复制，`AGENTS.md` 的开工前置第 2 条才能落地。
另一个动机是风险：v2 的 68 GB 当初没有第二份，如今已彻底找不回来（§39.1），这 2.13 GB 是同类隐患。

### 45.1 搬了什么

| 源（`tiger12:/pub/data/caohy`） | 目标（`tiger11:/pub/data/caohy`） | 体积 | 内容 |
|---|---|---|---|
| `logs/camra_hira` | 同名 | **2.13 GB / 14,362 文件** | 9 个 run、**1880 个 trial**（13,178 文件）、`quarantine` 9.3 MB、`assets/usd_cache` 18.8 MB、59 份运行日志 |
| `work/hira` | 同名 | **132 MB** | `CaMRA` 工作副本 11 MB、三个交付验证副本、`guard_check` 负冒烟、`run_*.sh` |

**未搬**：`camra_bundles`（v2 的 IT2999 包，HiRA 不用）、`fakehome`/`xdg*`（运行期缓存）、`IsaacLab` 与 `miniconda`（`.11` 自带）。

**路径布局两机一致**（都是 `/pub/data/caohy/...`），所以 `config/hira.env` **一个字都没改**；
采集用的 `--bundle` 默认指向仓库内的 `frozen/hira_go2_18500`（在共享 NFS 上，`.11` 本来就有）。
顺带核实：`CAMRA_HIRA_BUNDLE_ROOT=/pub/data/caohy/camra_hira_bundles` 这个变量在**两台机器上都是空指**
（目录从未存在），属历史遗留字段，迁移时**不要**去补建它。

### 45.2 传输方式：换了两次，原因要记

1. **`tar` 管道经操作机中转**（`ssh .12 'tar cf -' | ssh .11 'tar xf -'`）：可行但只有 **≈1 MB/s**
   （操作机到集群的带宽瓶颈），2.26 GB 需 40 分钟 —— 否决。
2. **两机直连**：`.11 → .12` 免密不通（共享 `~/.ssh/authorized_keys` 只有一把 ed25519，
   `~/.ssh/id_rsa.pub` 未列入）。**没有改动 `authorized_keys`，也没有复制私钥**。
3. **最终方案**：在 `.12` 起**临时只读 rsync 服务**（`port=8731`、`read only=true`、
   `hosts allow=192.168.20.11`、`hosts deny=*`），由 `.11` 主动拉取：
   **2.29 GB / 40 s，约 62 MB/s**。搬完按 pid 文件停服务、删除配置，**不留常驻进程、不改鉴权**。

> 踩坑记录：`pkill -f 'rsyncd_migrate.conf'` 会把**执行它的那个 shell 自己**也匹配掉，
> 表现是 SSH 会话 `rc=255` 直接断开（与 §43.4 的 L13 同一类自匹配问题）。正确做法是按 pid 文件 kill。

### 45.3 三重验收（都对上了才算搬成功）

| # | 检查 | 方法 | 结果 |
|---|---|---|---|
| 1 | **对冻结索引** | `tests/verify_migration_hira.py` 按 `delivery/bundle_manifest_hira_v1.json` 的 `sha256_tree` 逐 trial 重算 | **9/9 run、1880/1880 trial、usd_cache 7/7**，校验 2.13 GB，`verdict.all_matched=true` |
| 2 | **对源端逐字节** | `.11` 上 `rsync -a -c -n --itemize-changes` 反向比两棵树 | 两棵树均 **0 差异**（覆盖索引之外的日志、`quarantine`、工作副本） |
| 3 | **对锁** | `.11` 上 `DataLock.load(data_lock_hira_evaluation_v3.json, verify_hash=True)` + `verify_artifacts(/pub/data/caohy/work/hira/CaMRA)` | 返回**空列表 = 零项不符**（stage=evaluation、margins 仍 frozen） |

第 3 条是关键：它证明的不只是"文件齐了"，而是**资产链在新机上完整**——代码散列、α/δ、清单、地图全部解析得到。
报告：`reports/migration_tiger11_20261109.json`（+ `.sha256`）。

### 45.4 边界（下游必须知道）

1. **`.12` 上的原数据一个字节都没删**，现在是第二副本。要不要清理由用户决定；
   在任何一处被覆盖之前，**它就是唯一的旁证**，不要当垃圾顺手清掉。
2. **迁移不改变任何结论**：锁、α/δ、功率模型、期望值、清单散列**全部未动**，`§34–§37` 的结论照旧。
   本节只证明"同一批字节现在多了一个可用的位置"。
3. **解释器不同**：`.11` 的 `env_isaaclab` 是 Python **3.11.15**，`.12` 是 3.11.18。
   只影响运行时，不影响已落盘的数据；但**在 `.11` 上跑采集前**需按 §44.4 第 3 条确认 Isaac Lab 与 GPU 可用。
4. **验收只覆盖字节一致性**，不重跑能量/覆盖/几何；那些仍以 `§34–§38` 的报告为准。
5. `.12` 上保留 `tmp/rsyncd.log`（搬运日志）作证据，未删。

### 45.5 换机后怎么复核

```bash
cd /pub/data/caohy/work/hira/CaMRA/calibration/experiments/hira_go2_18500
/pub/data/caohy/miniconda/envs/env_isaaclab/bin/python tests/verify_migration_hira.py \
  --data-root /pub/data/caohy \
  --out reports/migration_tiger11_20261109.json
```

口径与 `build_delivery_manifest.py` 完全一致（单文件 `sha256(bytes)`；trial 级 `sha256` over
排序后的 `相对路径\0文件散列\n`，跳过 `__pycache__`）。任一 trial 不符返回码 1。

## 46. 本地文档同步与当前复核入口（2026-11-09）

本节追加在既有交接后，不覆盖 §1–45、原始报告、数据锁或40项计划状态。核对基线为主仓库 `35a7b4ab04881405bbbe4211f89eacf237ad6f78`。本次仅在 Windows 本地维护说明并进行静态检查，没有在服务器重跑仿真、复算原始轨迹或改变控制器。

- 根 README、AGENTS、各模块及文档索引已同步为 **40/40 项记录完成**；迁移状态按 §45 的报告更新为已完成，运行前仍须在 tiger11 核验资源与环境。
- [两套控制器结果与当前复核说明](../../../docs/controller_comparison_status_20261109.md)集中记录批间关节/上一动作复位、warmup 后名义初态对齐、全体业务分母和交付索引待复核事项。静态发现尚未量化对历史结果的影响；旧报告的数值复算一致不等于这些问题已被排除。
- v2 的 α/δ 为0.190483/2.613866 m，HiRA 为0.351568/9.032231 m。建议后续保留 v2 为开发基线、HiRA 为独立对照；这是建议，没有新实施切换或导航训练，也不是仅比较权重或宣称全面性能优劣。各自功率模型、足迹及裕度不得混用。
- 当前 HiRA 条件独立检验分母为377。保存覆盖计数按全体400条计算，成本/位置/联合为85.75%/84.50%/77.00%；未完成试验保留在分母，未计算新置信区间或覆盖原报告。
- 历史交付索引保留。当前 Git 字节复核发现换行、缺失 `env_min_start.log` 以及 `final_acceptance.json.sha256` 与索引不一致；本次后续文档修订也会与旧索引产生文档散列差异。复核原验收使用对应历史快照，不能重算旧锁或旧索引冒充文件未变。

下一步先核对批间状态和原始 pairing 首帧对齐，再决定新版本诊断、统计补齐或实验需求；本节不把这些后续事项计入旧任务的完成证据。

## 47. 用户确认后续固定使用 v2（2026-11-09）

用户已确认：后续开发固定使用 **`camra_go2_velocity_it2999_v2`**，作为硬性约束。该决定取代 §46 中尚属建议的控制器选择；不改写历史报告、数值、任务状态或冻结产物。

后续集成、校准复核及上层导航训练/评估须绑定 v2 的完整模型、资产、观测/动作、PD、时序及匹配校准配置。HiRA 策略、适配代码和实验资料仅保留为历史参考/对照；未经用户明确授权，不切换控制器或 checkpoint，不重新训练替换 v2，不默认开展新 HiRA 实验。缺 v2 资源先获取核验，不以 HiRA 顶替。

HiRA 的独立待复核事项继续留存，不默认成为 v2 开发前置任务；共享代码问题按对 v2 的实际影响核查。v2 的低速局限、旧验收失败与历史校准结果保留。完整约束见[根 AGENTS](../../../../../AGENTS.md)，当前结果与选择记录见[对比说明](../../../docs/controller_comparison_status_20261109.md)。此次仅同步文档，没有运行控制器或修改代码与配置。

## 48. 当前转入固定 v2 导航计划（2026-11-09）

当前入口为[60项上层导航计划](../../../../../doc/plan/20261109_camra_navigation_implementation.json)，版本1.2.0，0/60项完成，首项 NAV-001。先学习背景和参考代码，再核验 tiger11（192.168.20.11）、v2及配套校准资源；随后按依赖完成名义环境、底层接入、算法验证、名义训练、预算与clean闭环、固定策略扰动检验及复现交付。当前仅维护文档和计划，尚未启动导航实现或服务器实验。

计划 `upstream_correction_policy` 允许既定范围内回修前置实现，修正主体记录追加到发现问题的当前任务；原记录和证据保留，受影响验收与门禁重验后才能继续。该规则不授权修改 HiRA 参考仓库、冻结 v2、旧校准或论文。§46中的复核建议保留为历史记录，不默认成为 v2 导航的前置实验；HiRA 的40项完成状态及原始报告、锁均不变。

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

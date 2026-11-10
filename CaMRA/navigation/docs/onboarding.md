# 仓库与授权基线（NAV-001）

采集时间：2026-11-09 23:45（+08:00）。本文件只记录阅读与核对结果，不含实现。

## 1. 快照

| 项 | 值 |
|---|---|
| 主仓库 HEAD | `8eddd8fc16f212554d7d833ad0bb816b3cc871ef`（`docs: use canonical repository URL after GitHub migration`） |
| 分支 | `main`，跟踪 `origin/main` |
| 计划创建时记录的 HEAD | `35a7b4ab04881405bbbe4211f89eacf237ad6f78`（本计划 `source_snapshot`） |
| HEAD 差异原因 | 计划创建后主仓库新增 2 个提交：`3108413` 固定 v2 导航计划与交接、`8eddd8f` 仓库 URL 规范化。两者均为文档提交，未改代码与冻结产物 |
| HiRA 子模块 | `6a05af55521e5f6668c50e9f90cf7049b3e35cd9`（`heads/master`），与计划锁定提交一致 |
| 工作树改动（tracked） | 无 |
| 未跟踪文件 | 无（本文件写入前） |

## 2. 工作树与 dirty 归属

`git status --porcelain -uno` 为空，`git ls-files --others --exclude-standard` 为空；`git -C HiRA status --porcelain -uno` 为空。

结论：本次开工前没有任何未归属改动，不需要保留或恢复他人工作。本任务新增的 `CaMRA/navigation/docs/onboarding.md` 是本任务产物，属新增而非修改既有内容。按 NAV-001 约束，本次未执行 `git pull` / `reset` / `clean`。

## 3. 用户硬约束（2026-11-08 / 11-09 确认，不可自行更改）

1. 后续开发、集成、校准复核及上层导航训练/评估，**整套固定** `camra_go2_velocity_it2999_v2`；不换 HiRA、不改其他 checkpoint、不重训。
2. 权重、资产、观测/动作映射、PD、频率、功率模型、足迹、α/δ、数据锁必须**配套同源**，不能交叉搭配。
3. 执行主机固定 **tiger11 / 192.168.20.11**；`.10` 禁用，登录失败也不许自动转 `.12`。
4. 服务器终端任务全部在 **tmux** 内执行并留存会话/命令/日志。
5. **论文只读**：`manuscripts/**`、`*.tex`、`*.bib`、投稿源与 PDF、`docs/paper*`、`submission*` 禁止写入，不编译。
6. 首轮只做两种方法：无校准（α=0、δ=0，仍含足迹）与完整方法；不默认增加消融或外部基线。
7. 随机物理环境下再训练不属于首轮必做项。

## 4. 阅读清单（本任务实际读过）

- 根 `AGENTS.md`（当前任务、执行主机、编辑边界、论文禁令）
- 根 `README.md`、`CaMRA/README.md`、`CaMRA/navigation/README.md`、`CaMRA/locomotion/README.md`、`CaMRA/calibration/README.md`
- `doc/README.md`、本计划全文（document / background / goal / scope / global_constraints 24 条 / uncertainty_policy / runtime_contract / method_contract / map_contract / evaluation_contract / reference_map 13 条 / paths / gates 8 条 / 60 项 tasks / final_acceptance 15 条 / upstream_correction_policy）
- `CaMRA/calibration/docs/controller_comparison_status_20261109.md`
- 技能 `create-plan-worklog` 的 SKILL 与 `references/execution-rules.md`

## 5. 现状盘点

| 模块 | 实际状态 |
|---|---|
| `CaMRA/navigation/` | **仅有 README**；无环境、网络、预算搜索、训练或评估代码。计划的 `proposed_*` 路径均为拟建 |
| `CaMRA/locomotion/` | v2 已冻结；另有 HiRA 适配（历史对照，本轮不作主线） |
| `CaMRA/calibration/` | v2 与 HiRA 双套工具与结果；本轮只读复用，不改锁、不反调 α/δ |
| `RC-PPO/` | 只读算法参考（JAX 实现，主入口 `EC-EFPPO` / `EFPPO`） |
| `HiRA/` | 只读子模块，本轮仅借鉴分层结构 |
| `manuscripts/` | 只读 |

## 6. 资源位点：已核验与待核验

**已在 tiger11 实际核验（2026-11-09 迁移，属 NAV-009 的前置事实，尚未走验收流程）**

- v2 冻结包 `/pub/data/caohy/camra_bundles/frozen/camra_go2_velocity_it2999_v2`：29/29 文件散列一致，tree `d0f0ec3c…` 复现一致。
- Go2 资产 `/pub/data/caohy/unitree_model/Go2`：4 文件齐全，抽样 usd 散列与源一致。
- 训练权重 `/pub/data/caohy/logs/rsl_rl/rsl_rl/camra_go2_locomotion/2026-11-02_11-14-49_formal_s0_3000it_r4/model_2999.pt`：散列一致。
- v2 校准日志 `/pub/data/caohy/logs/camra`：2168 文件、3,262,425,965 字节与源逐字节一致。
- 隔离环境 `env_isaaclab_v0472`：isaaclab 0.47.2 / assets 0.2.3 / tasks 0.11.6 / rl 0.4.4 / rsl-rl-lib 3.0.1 / onnxruntime 1.31.1 / numpy 1.26.0 / isaacsim 5.1.0.0；源码在 `/pub/data/caohy/IsaacLab_v0472`。
- 运行期目录 `fakehome`/`tmp`/`xdgdata`/`xdgcache`/`xdgconfig` 已建。

**尚待 NAV-007–010 逐项核验并登记**

- `runtime_paths.json` 尚未建立；计划 `runtime_contract.python_hint` 仍指向 `env_isaaclab/bin/python`（isaaclab 0.54.2），与实际应使用的 `env_isaaclab_v0472` 不一致，须在 NAV-008 正式裁定。
- GPU 空闲度、tmux 会话、依赖完整性、拟建导航目录均未登记。
- 计划 `proposed_navigation_work_root` / `proposed_navigation_output_root` 尚不存在。

## 7. 边界

- 本文件不含任何实现或运行承诺；`proposed_*` 路径不代表已实现。
- 已核验资源属于环境事实，不构成 NAV-009 的验收结论；NAV-009 仍需按自身验收标准重新确认并写记录。
- 历史任务完成状态、HiRA 40/40 记录、v2 旧验收失败与低速局限均不因本文件而改变。

---

# 方法理解与边界（NAV-002）

依据：`manuscripts/CCDC/latex/1029/main.tex` §3–4（只读）、`RC-PPO/README.md`、`calibration/src/camra_calibration/{step_cost,power}.py`、`config/power_model_v1.json`。

## 1. 一回合的完整因果链

```
策略 π(u | o, y, z)                    ← 输入含安全历史 y、剩余预算 z
   │  机身系速度 u_t = [vx, vy, wz]，10 Hz
   ▼
名义转移 x_{t+1} = f̄_cl(x_t, u_t)      ← 理想跟踪下按命令做平面精确积分
   │
   ▼
名义步成本 c_nom = P̂_nom(u_t) · 0.1 s  ← 单位 J；冻结模型 history_k=0，只用当前命令
   │
   ▼
预算扣减 z_{t+1} = z_t − (1+α)·c_nom   ← 支出为正；α 只作用在成本侧
   │
   ├──→ 安全侧：F_δ = {x : d(x,F) ≤ δ}，y_{t+1} = max{ I(x_{t+1} ∈ F_δ), y_t }（不可逆）
   │
   ▼
增广目标 ĝ(x,y,z) = max{ g(x), C·y, −z }   → 到达 ⇔ ĝ ≤ 0
   │
   ▼
学习目标 W = max{ V_h , V_c − z/z_ref }    ← 双评价网络；actor 最小化该残差
   │
   ▼
物理执行：u_t 送入冻结 v2（策略 50 Hz / 物理 200 Hz / 命令 10 Hz）
   │
   ▼
真实机械功 J_exec = Σ_j Σ_k max(τ_j·q̇_j, 0) · 0.005   ← 单独记账，不回写内部预算
```

两个裕度各管一件事，且**只能各用一次**：

- `α` 只改成本：`c → (1+α)c`，作用在预算扣减与成本评价网络。
- `δ` 只改安全：危险集合膨胀 `obstacle ⊕ footprint ⊕ disc(δ)`，作用在安全历史与到达判据。

关键细节：`z_ratio` 的分母是**本回合初始预算**且下限截 0，不能直接当成固定 `z_ref` 的归一化；而 `z_ref` 是场景参考预算，两者必须显式区分并做逆变换测试（计划 `method_contract.budget_semantics`）。

## 2. 三种"执行"必须分开

| 类型 | 是什么 | 有什么误差 | 用途 |
|---|---|---|---|
| **确定性名义** | 纯数学积分 `x_{t+1}=f̄_cl(x_t,u_t)`，无物理引擎 | 无（但只是抽象） | 上层训练与环境验证 |
| **clean 物理** | Isaac 仿真，关闭随机扰动 | **仍有**模型—执行偏差：接触、摩擦、力矩限幅、采样相位 | 部署策略的首次物理检验 |
| **扰动 物理** | clean + friction/mass/push 等预登记扰动 | 上述偏差 + 扰动 | 固定策略的鲁棒性检验 |

常见误解：把 clean 当成"无误差"。关闭扰动只去掉随机源，不消除系统性偏差。

## 3. 适用性缺口（原校准 ≠ 未来导航）

原校准的测量契约是：**同一命令程序、1 秒零命令 warmup、固定 30 秒窗口、只在窗口起点对齐一次**。导航任务与它有三处本质差别：

1. **终止方式不同**。校准跑满 30 秒；导航"到达即停"，会提前终止。提前终止的能量是**前缀**，与完整 30 秒窗口的成本不是同一个量，分母不能混用。
2. **指令分布不同**。校准用分段命令程序（含阶跃与 0.2 秒过渡）；闭环导航是 10 Hz 策略反馈，指令变化率、切换频率、低速停留比例都不同。δ 与 α 是在前者上测的分位数，不能默认覆盖后者。
3. **几何量含义不同**。`δ` 是平面位置最大偏差的经验分位数，**不含航向误差**；而足迹随姿态变化，矩形足迹 + δ 对旋转不安全（计划因此要求 NAV-057 给出姿态处理或外接圆保守近似）。

由此推出计划中三条硬性安排：G2A（训练前用闭环指令和旧裕度做限额 pilot）、NAV-034/043（对短训练策略与部署策略分别复查）、NAV-048（最终审计）。**pilot 通过不等于旧 30 秒覆盖结论可以继承**。

## 4. 本轮不做的事

- 不把论文里的 a.u. 代理成本、示例数值或 5 Hz 历史配置迁进当前 J 成本与 10/50/200 Hz 契约。
- 不把 PPO 损失下降、进程退出码 0、模型能加载、校准覆盖率当成收敛或导航成功的证明。
- 不声称经验覆盖构成随机安全保证。

## 5. 待裁定项（留给后续任务，不在本任务解决）

- 速度域冲突：`power_model_v1.domain` 记录 `|vx|≤1.0, |vy|≤0.4, |wz|≤1.0`，而 `margins_v1.applicability` 写 `2/1/2`。两者不一致，须由 NAV-055 依据原始 applied 命令与数据锁裁定，旧文件不改。

---

# v2 来源契约（NAV-004）

依据：`locomotion/docs/frozen/camra_go2_velocity_it2999_v2.frozen_manifest.json`、
`calibration/config/{calibration_protocol_v1.0.2,power_model_v1,footprint_v1,margins_v1,data_lock_v1,step_cost_schema_v1}.json`、
`calibration/README.md`、`calibration/docs/implementation_handoff.md`、`controller_comparison_status_20261109.md`。全部只读。

## 1. 绑定清单（必须同源，不可交叉搭配）

| 层 | 对象 | 关键值 / 绑定方式 |
|---|---|---|
| 权重 | `model_2999.pt`（`logs/rsl_rl/.../2026-11-02_11-14-49_formal_s0_3000it_r4`） | 由冻结包 `integrity.files` 29 项散列绑定 |
| 网络 | ActorCritic 45→512→256→128→12，ELU，torch.no_grad 推理 | `training.actor`，参数总量 384,921 |
| 资产 | `unitree_model/Go2`（4 文件） | `assets.go2_tree.tree_sha256 = 95605072…` |
| PD | kp 25.0 / kd 0.5 / friction 0.01 | `contract.robot.gains` |
| 观测 | 45 维：`base_ang_vel×0.2` → `projected_gravity×1` → `velocity_commands×1` → `joint_pos_rel×1` → `joint_vel_rel×0.05` → `last_action×1`，统一 clip ±100 | `contract.observation` |
| 动作 | `JointPositionAction` 12 维，`scale 0.25`，`offset=default_joint_pos`，wrapper 不裁剪 | `contract.action` |
| 关节顺序 | 按类型分组：`FL/FR/RL/RR hip` → `thigh` → `calf` | `contract.action.joint_names` |
| 时序 | sim 0.005 s / decimation 4 / step 0.02 s / 策略 50 Hz / 命令 10 Hz | `contract.timing` + `protocol.clocks` |
| 命令域 | `vx∈[-1,1]`、`vy∈[-0.4,0.4]`、`wz∈[-1,1]`，先裁剪后记为 applied | `contract.command` + `protocol.command_domain` |
| 功率模型 | `power_model_v1.json`（power-v1，`history_k=0`，`step_cost_dt=0.1`） | `data_lock_v1.power_model.content_sha256 = bfe3b159…` |
| 足迹 | `footprint_v1.json`（凸多边形，面积 0.45796 m²，`base_link_xy`，**不含 δ**） | `data_lock_v1.footprint.content_sha256 = a11a3572…` |
| 裕度 | `margins_v1.json`：α=`0.19048279965649906`，δ=`2.6138664115280967` m，n=790 | `data_lock_v1.margins.content_sha256 = dd76972d…` |
| 步成本 | `step_cost_schema_v1.json`，只接受 `unit="J"` | `data_lock_v1.step_cost_schema.content_sha256 = a2a5fbc6…` |
| 协议 | `calibration_protocol_v1.0.2.json`（frozen，`supersedes 1.0.1`） | `data_lock_v1.protocol.baseline_bound.sha256` |
| 数据锁 | `data_lock_v1.json` | 自绑定 code/protocol/model/footprint/margins/maps/trial_lists |

## 2. 三类“数值”不要混用

| 类型 | 例子 | 用法 |
|---|---|---|
| **精确文件值** | `alpha = 0.19048279965649906` | 实现与报告中直接加载，不得手抄近似值 |
| **报告近似值** | 文档中的 `α ≈ 0.190483`、`δ ≈ 2.6139` | 只用于阅读与尺度理解，不能进代码或锁 |
| **内容 hash** | `margins_v1.sha256 = dd76972d…`（对文件内容） | 绑定“文件未被修改” |
| **原始字节 hash** | `integrity.files` 中逐文件 sha256；`tree_sha256` 对 `"rel sha bytes"` 排序后 `\n` 连接 | 绑定“字节未被修改”，换行差异也算变化 |

注意：`data_lock` 里存的是 **content_sha256**，`frozen_manifest.integrity` 存的是**逐文件 sha256 + tree 散列**。两者口径不同，核查时不能互相替代。

## 3. 速度域：三方一致，只有一处文本不一致

| 来源 | vx | vy | wz |
|---|---|---|---|
| `calibration_protocol_v1.0.2.command_domain` | ±1.0 | ±0.4 | ±1.0 |
| v2 `contract.command.ranges` | ±1.0 | ±0.4 | ±1.0 |
| `power_model_v1.domain`（拟合样本极值） | ±1.0 | ±0.4 | ±1.0 |
| **`margins_v1.applicability.command_domain`（纯文本）** | **±2.0** | **±1.0** | **±2.0** |

结论（仅记录，不裁定）：协议、控制器契约、功率模型**三方一致**为 `1.0/0.4/1.0`；只有 `margins_v1` 的 applicability 描述文字写成了 `2/1/2`。计划 `background.known_boundaries` 已把此项列为 NAV-055 待裁定，旧文件不得修改，最终以原始 applied 命令抽样与数据锁为准。

## 4. 服务器资源线索（详情见 `/pub/data/caohy/ENV_v0472_README.md`）

- 仓库：`/home/caohy/repositories/Paper`（`/home` 为共享 NFS，可见 ≠ 在正确节点执行）
- 冻结包：`/pub/data/caohy/camra_bundles/frozen/camra_go2_velocity_it2999_v2`（29/29 散列已复验）
- 权重：`/pub/data/caohy/logs/rsl_rl/rsl_rl/camra_go2_locomotion/2026-11-02_11-14-49_formal_s0_3000it_r4/model_2999.pt`
- 资产：`/pub/data/caohy/unitree_model/Go2`
- 校准日志：`/pub/data/caohy/logs/camra`（2168 文件 / 3,262,425,965 字节已逐字节核对）
- 解释器：`/pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python`（isaaclab 0.47.2）
- 环境配置模板：`/pub/data/caohy/v2_navigation.env`

## 5. 旧起点/复位问题：对 v2 共享代码的核查点

`controller_comparison_status_20261109.md` 记录了 HiRA 侧 5 项静态问题。它们是 **HiRA 的** `adapters/hira_go2/{collector,env_runner}.py` 发现，不能直接推定为 v2 也存在，但必须核查共享/相似的采集链路：

| 待核查点 | HiRA 的发现 | v2 侧需检查的位置 |
|---|---|---|
| 批间初态是否完整复位 | `execute_batch` 只重写 root 位姿/速度，未复位关节与上一动作 | `calibration/scripts/collect_trials.py`、`calibration/sim/recorder.py` |
| warmup 是否真跟踪零速 | `env_runner.warmup` 用零动作默认角 PD，非策略零速指令 | 同上，确认 1 s warmup 的命令来源 |
| 测量起点用实测还是预设 | 元数据存 `spec.initial_pose`，脚本用该预设重建名义轨迹 | `evaluate_heldout.py`、`nominal.py` 的起点来源 |
| 全体分母是否齐备 | 覆盖率用“完整且可算”子集，协议要求同时报全体业务试验 | `protocol.statistics` / 报告分母字段 |
| 交付索引可移植性 | CRLF 与索引构建时序导致 3/231 不一致 | `locomotion/scripts/artifact_index.py verify` |

**本轮不做**：不修复、不重跑、不改旧锁。仅登记为 NAV-024–028 与 NAV-048 的核查输入。

## 6. 本轮禁止动作

- 不重新训练、不筛选其他 checkpoint、不切换低层；不改 `margins_v1` / `power_model_v1` / `footprint_v1` / `data_lock_v1` 的任何字节。
- 不反调 α/δ，不重做校准，不用新数据覆盖旧锁；需要新版本时另建文件并重新散列。
- 不改写历史任务状态与 HiRA 40/40 记录；`TASK-036 overall_pass=False` 与低速局限保持原样。
- 不把校准覆盖率当作导航成功率或随机安全保证。

---

# 六阶段实施理解与代码复用矩阵（NAV-006）

综合 NAV-001–005 的源码证据（见本文各章节与 `source_reference_map.md`）。
“六阶段”对应 `AGENTS.md` 的实施顺序，复现交付作为收尾；与计划 `execution_focus.stages` 的映射见表末。

## 1. 复用矩阵

| # | 阶段 | 现成可复用（直接调用） | 只参考结构（需重写） | 必须新写 | 拟验收证据 |
|---|---|---|---|---|---|
| 1 | 统一契约与确定性名义环境 | `calibration_protocol_v1.0.2`（时钟/命令域/终止）、`step_cost_schema_v1`、`power_model_v1`、`footprint_v1`、`margins_v1`、`data_lock_v1` | RC-PPO `wind_field` 的 `step_env`（名义步进结构） | 任务/事件/终止契约；观测与坐标契约；确定性名义动力学 `f̄_cl`；J 成本与有符号预算适配；足迹+δ 安全判据；宽通道地图生成器；膨胀后通路与 30 s 可达检查；训练/开发/选择/评估互斥清单 | 契约 JSON + 手算短轨迹对照 + 地图通路图 + 名义环境单元测试 |
| 2 | 提前接通 v2 并验证时序/复位/计量 | `rollout_exported_policy.py`（加载与闭环）、`external_command.py`（10 Hz 零阶保持）、`recorder.py`（记录范式）、v2 冻结包 `contract` | HiRA `hierarchical_go2_env.step` 的子步循环写法 | Isaac Lab 侧的 v2 加载适配器；10/50/200 Hz 桥；完整复位与 warmup 起点；200 Hz 真实机械功记录 | 6000/1500/300/301 计数实测；复位前后状态对比；机械功与离线复算一致 |
| 3 | 原算法数值与短训练验证 | —— | RC-PPO `gae.py`（`calculate_indexs3` / `calculate_gae2` / `calculate_gae_reach4`）、`EFPPO_utils._ecefppo_update`、`actocritic` 网络 | 双评价目标与停止分段（PyTorch）；成本最小化 PPO；轨迹缓冲与可恢复训练循环；可枚举小问题对照 | 手算短轨迹逐项对齐；梯度方向检查；小批 loss 曲线；与独立实现数值一致 |
| 4 | 名义训练与部署策略冻结 | 数据锁/散列工具（`build_data_lock.py` 模式） | RC-PPO 的训练循环组织 | 无校准与完整方法两组训练；确定性部署策略；critic 在固定 actor 下重拟合；checkpoint 冻结与清单 | 训练日志 + 预登记配额 + 冻结清单 + 选择规则记录 |
| 5 | 预算选择与 clean 闭环 | `power.py`（`PowerModel`）、`step_cost.py`（`BudgetState`）、`evaluate_heldout.py`（覆盖率口径与分母写法） | `root_finding.Bisection`（需补端点/单调性/无解） | J 单位预算搜索；独立选择集；在线闭环；正式评估数据锁与对照 | 可行/无解比例；搜索诊断；clean 轨迹与复算 |
| 6 | 固定策略扰动检验 | `sim/profiles_sim.py`、`profiles.py`（五 profile 定义） | —— | 扰动注入与固定策略评估；前缀/完整窗口分母；独立复算与统计区间 | 全体分母报告；失败台账；区间由真实轨迹复算 |
| — | 复现交付 | 冻结包与清单模式、`verify_delivery_bundle.py` | —— | 独立目录复现包；审计；交接 | 独立目录重载；逐项对账 |

计划 `stages` 映射：阶段 1→NAV-012–023；阶段 2→NAV-024–028；阶段 3→NAV-029–035；阶段 4→NAV-036–039；阶段 5→NAV-040–045；阶段 6→NAV-046–050；收尾→NAV-051–054。

## 2. 关键判断（来自源码，不是摘要）

1. **成本与预算必须新写适配层，不能直接复用 RC-PPO 的 energy**。原实现用归一化 `[-1,1]`（×400）且成本是 `sum(control²)/2`；CaMRA 要 J 单位、`(1+α)` 扣减、固定 `z_ref` 归一化。
2. **算法不能"翻译"而必须"重实现 + 独立对照"**。原实现是 JAX + `max` 损失 + 梯度下降；直接改写成 PyTorch 常见 PPO 形态会符号反掉。
3. **分层调用结构可借鉴，复位隔离必须自己补**。HiRA 的子步循环在本 env 复位后仍继续跑，CaMRA 必须隔离。
4. **v2 接通要早于训练**。计划把 v2 放在阶段 2 而非最后，因为时序/复位/计量错误会在训练后被误读成"算法不行"。
5. **校准产物只提供数值，不提供结论**。α/δ 是分位数，覆盖率是经验频率，二者都不是导航成功率或安全保证。

## 3. 实际未决点与解除条件

| # | 未决点 | 现状 | 解除条件 | 负责任务 |
|---|---|---|---|---|
| 1 | 速度域到底是 `1.0/0.4/1.0` 还是 `2.0/1.0/2.0` | protocol、v2 contract、power_model 三方为 1/0.4/1；仅 `margins_v1` 文本为 2/1/2 | 抽样原始 applied 命令极值 + 数据锁核对，形成唯一裁定写入导航契约 | NAV-055 |
| 2 | 上层实现框架（PyTorch 还是沿用 JAX） | 未见授权变更；计划 `planned_decisions` 标注“未假定已获准替换原 JAX 实现” | 列出兼容方案及对数值对照/成本的影响，用户确认后冻结 | NAV-008 |
| 3 | 解释器路径冲突 | `runtime_contract.python_hint` 指向 0.54.2 环境；实际应为 `env_isaaclab_v0472`（0.47.2） | 在 `runtime_paths.json` 中登记并说明裁定依据 | NAV-008 / NAV-009 |
| 4 | 足迹的姿态处理 | `footprint_v1` 是二维凸多边形，不含航向 | 给出外接圆或有依据的姿态处理，并附保守性论证 | NAV-057 |
| 5 | `z_ref` 的物理定义 | RC-PPO 用 `x*variance+mean` 反归一化，不是 J | 定义 `z_J`/`z_ref_J` 映射并做逆变换测试 | NAV-056 |
| 6 | 正式种子、试验数、GPU 小时 | 计划明确“不在计划中虚构已确定配额” | 出示资源表并取得授权后预登记 | NAV-021 |
| 7 | 地图净宽与 30 s 可达 | 7–8 m 是初始设计，不是安全证明 | 膨胀后通路 + 转速 + 时限检查通过 | NAV-017 |

## 4. 进入下一阶段前必须满足的条件

- 上述未决点 1、3 必须在 G0（NAV-011）前关闭；2 在冻结前关闭；4、5 在 G1（NAV-023）前关闭；6 在正式训练前登记；7 在 G1 前给出证据。
- 本阶段（NAV-001–006）只完成学习与文档，未安装依赖、未运行仿真、未创建任何实现代码。
- 服务器侧已有 v2 资源与 0.47.2 环境属**事实**，但尚未按 NAV-009 的验收标准走流程，不能视为该任务已完成。

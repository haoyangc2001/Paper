# CaMRA Locomotion 实施交接笔记

本文件是 `doc/plan/20261031_camra_locomotion_implementation.json` 执行过程中的**集中交接笔记**，
承载论文到代码的映射、仓库差异、接口约定、运行环境与复现入口。逐任务的执行证据写在计划 JSON 的
`work_records` 中，不在此另建过程文件。

## 目录索引

| 章节 | 内容 | 由哪个任务产出 |
| --- | --- | --- |
| 1. 执行环境与基线 | 主机、tmux 会话、Git 基线版本 | TASK-001 |
| 2. 仓库地图与读写边界 | 各目录作用、体积、可否修改 | TASK-001 |
| 3. 三个模块现状 | locomotion / calibration / navigation | TASK-001 |
| 4. 既有产物与源码的区别 | 上游示例模型 vs 本项目训练产物 | TASK-001 |
| 5. 稿件位置 | 1029 当前稿、1020 历史稿 | TASK-001 |
| 6. 参考仓库来源 | HiRA、unitree_rl_lab、RC-PPO | TASK-001 |
| 7. 论文到代码映射 | 概念→模块→文件 | TASK-002（待填） |
| 8. 观测/动作/频率契约 | 45 维、12 关节、200/50/10 Hz | TASK-009（待填） |
| 9. 评估与导出要求 | 导出路径、数值一致性 | TASK-011（待填） |
| 10–29 | 逐任务追加的正文（论文核对、代码阅读、接口实现、训练与评估），未逐条列入本表 | TASK-003–TASK-022 |
| 30. 阶段一交付与接手须知 | 交付物与校验入口、真实命令、必须保持一致的五件事、没做的事、踩过的坑 | TASK-028 |
| 31. 底层验收证据复核 | 证据索引、训练模型数、变体覆盖与指标完整性的只读核对 | TASK-029 |
| 32. 变体覆盖修复 | 逐变体配额调度、variant 日志、步数保护后的未完成判定 | TASK-031 |
| 33. 有效数与完整性判定 | 逐指标有效数/原因、缺失即证据不足、跨种子完整性 | TASK-032 |
| 34. 停车概率渐增的时间单位 | 控制步/迭代/仿真秒的换算、注释纠错、行为不变 | TASK-033 |
| 35. 仓库内复现入口与产物索引 | repro/* 入口、路径配置化、产物索引与预检三态判定 | TASK-034 |
| 36. 回归测试、接口契约与小批验证 | 88 项回归、9 项频率契约、45 回合逐变体核对、810 项复算 | TASK-035 |
| 37. 确认性复测 9000 回合 | 逐变体判定、旧覆盖只有 10/15 变体、单方向精度差 3.5%、未通过 | TASK-036 |
| 38. 收尾与未关闭项 | 文档改动清单、干净克隆复核、v2 包未变原因、计划状态与校准前置条件 | TASK-037 |
| 39. 低速工况的稳态缺口 | 裁决②诊断：bias/std 分解、时间分箱、跨场景对照，弱区是低速而非侧向 | TASK-036 追加 |
| 40. 速度扫描与起步死区 | 16 点扫描：≤0.20 m/s 完全不动、>0.20 固定缺口 0.09；推翻精度假设 | TASK-036 追加 |
| 41. 命令采样 A/B | 继续训 250 迭代已让 −0.30 的 rmse 从 0.103 降到 0.060；去站立采样再撬开低速死区 | TASK-036 追加 |
| 42. 延长训练与判断修正 | 再训 500 迭代：3248 的改善未保持，死区全程未动，改采样才是有效方向 | TASK-036 追加 |

---

## 1. 执行环境与基线

| 项 | 值 |
| --- | --- |
| 执行主机 | `tiger12.ios.ac.cn`（192.168.20.12），Ubuntu 24.04.5，96 核 / 503G |
| 工作区 | `/home/caohy/repositories/Paper`（共享 NFS） |
| tmux 会话 | `camra-locomotion`，窗口按任务命名（`TASK-xxx`），tmux 3.4 |
| 日志根目录 | `/pub/data/caohy/logs/camra/<TASK-ID>/`（节点本地盘） |
| 计划文件 | `doc/plan/20261031_camra_locomotion_implementation.json` v1.2.0 |
| 校验器 | `/pub/data/caohy/tools/validate_worklog.py`（`--mode update`） |

**Git 基线**（2026-10-31 实测）：

| 对象 | 版本 |
| --- | --- |
| 根仓库 | `main` @ `8bed6d58b94e03111a0ec5d2fead49bcfe53f147` |
| HiRA 子模块（gitlink） | `6a05af55521e5f6668c50e9f90cf7049b3e35cd9`（与计划 baseline 一致） |
| unitree_rl_lab 上游快照 | `4960b84732b0c2ec593dccbfe963fda1bcd7b1e3`（2025-11-19） |

`.gitmodules` 与 gitlink 已核对：`HiRA` 路径/URL 正常，子模块 `heads/master` 干净、无游离改动。
仓库内**无 `AGENTS.md`**（本项目未使用该约定文件）。

## 2. 仓库地图与读写边界

| 路径 | 体积 | 作用 | 边界 |
| --- | --- | --- | --- |
| `CaMRA/locomotion/` | 29M（174 文件） | Go2 底层速度跟踪（宇树官方框架快照） | ✅ **可编辑**，本阶段主战场 |
| `CaMRA/calibration/` | 1 文件 | 执行误差采集与 α/δ 裕度校准 | ⏸ 仅 README，本阶段不实现 |
| `CaMRA/navigation/` | 1 文件 | 上层最小成本到达避障 | ⏸ 仅 README，本阶段不实现 |
| `HiRA/` | 1.3G | Reach-Avoid 分层导航参考（Git 子模块） | ⛔ **只读**，锁定 `6a05af5` |
| `RC-PPO/` | 1.5M | 原算法冻结参考实现（JAX / EC-EFPPO） | ⛔ **只读**，不修改 |
| `manuscripts/` | 1.1G | 论文、图、数据、投稿材料 | ⛔ **只读**，本阶段不改 |
| `doc/plan/` | 84K | 本计划 JSON | ✏️ 仅按技能追加 work_records |

## 3. 三个模块现状

- **locomotion**：已完整引入上游源码快照（161 个受版本控制文件，逐文件 SHA-256 与上游一致），
  仅 `README.md` 改名为 `README.upstream.md`、新增本地 `README.md` 与 `UPSTREAM.md`。
  Python 包名 `unitree_rl_lab`、脚本 `unitree_rl_lab.sh`、任务 ID `Unitree-Go2-Velocity` 保留上游命名。
  **当前状态：尚未在本项目内训练或产出 Go2 策略。**
- **calibration**：只有用途说明 README，无代码。
- **navigation**：只有用途说明 README，无代码。

## 4. 既有产物与源码的区别

`find` 仅发现 3 个 `policy.onnx`，全部位于 `CaMRA/locomotion/deploy/robots/g1_29dof/`：

```
deploy/robots/g1_29dof/config/policy/velocity/v0/exported/policy.onnx
deploy/robots/g1_29dof/config/policy/mimic/dance_102/exported/policy.onnx
deploy/robots/g1_29dof/config/policy/mimic/gangnam_style/exported/policy.onnx
```

**这些是上游附带的 G1 示例模型，不是 Go2 策略，也不能作为本项目任何训练结论的证据。**
仓库内**没有任何 Go2 检查点、训练日志或导出产物**（无 `*.pt`、无 `logs/` 目录）。

## 5. 稿件位置

| 稿件 | 路径 |
| --- | --- |
| 当前稿 1029 | `manuscripts/CCDC/latex/1029/`（主文件 `conference_101719.tex`） |
| 历史稿 1020 | `manuscripts/CCDC/latex/1020/` |
| 1020 带附录版本 | `manuscripts/CCDC/latex/1029/archive/main_1020_with_appendices.tex` |
| 模板 | `manuscripts/CCDC/latex/template/` |

历史稿仅作为**历史实现设定**参考，不得当作当前可执行配置或训练达标证据（详见 TASK-003）。

## 6. 参考仓库来源

| 参考对象 | 来源 / 版本 | 用途 |
| --- | --- | --- |
| HiRA | `haoyangc2001/HiRA` @ `6a05af5` | 上下层衔接、频率经验、已有 Go2 策略与轨迹评估工具的**参考**，非本项目成果 |
| unitree_rl_lab | `unitreerobotics/unitree_rl_lab` @ `4960b847` | 底层训练/部署框架基线 |
| RC-PPO | 本地冻结副本，无独立上游元数据 | 原最小成本到达避障算法的阅读与对照，不直接改 |

RC-PPO 目录名为本地沿袭，内部训练入口命名为 `EC-EFPPO` / `EFPPO`；因缺少上游元数据，
**无法凭目录名确认与论文的逐项一致性**，只能作为参考实现阅读。

## 7. 论文到代码映射（TASK-002）

论文正文 `manuscripts/CCDC/latex/1029/main.tex`（693 行，6 节）：

```
71  Introduction
144 Related Work
192 Problem Formulation（MCRA/RC-PPO 回顾、四足闭环执行、部署目标）
259 Execution-Calibrated CaMRA（系统总览、冻结速度接口、失配校准、鲁棒 MCRA）
377 Experiments（设置/协议、整体性能、校准与裕度分析、预算选择、扰动、真机）
673 Conclusion
```

> 注：同目录 `conference_101719.tex` 是 IEEEtran 模板样例，不是本文正文。

| 论文概念 | 定义位置 | 对应模块 / 代码 | 本阶段状态 |
| --- | --- | --- | --- |
| 冻结速度接口（pretrained frozen tracker） | §IV-B | `CaMRA/locomotion/`，任务 `Unitree-Go2-Velocity` | 代码具备，**尚未训练** |
| 低层输入：本体感知 + 机身速度命令 `u=[vx*,vy*,ωz*]` | §IV-B | 45 维策略观测 | 结构已固定 |
| 低层输出：关节命令 | §IV-B | 12 关节位置增量，PD 25/0.5 | 已核对 |
| 名义闭环抽象 `f̄_cl`（理想跟踪下积分命令速度） | §III-B、§IV-B | 仅论文公式，无代码 | 本阶段不实现 |
| 执行误差 `ε_c`（相对代价超支）、`ε_d`（最大平面偏差，式 2） | §IV-C | `CaMRA/calibration/`（仅 README） | 本阶段不实现 |
| 校准裕度 `α=Q_pc(ε_c)`、`δ=Q_pd(ε_d)`（式 3） | §IV-C | 同上 | 本阶段不实现 |
| 校准后代价 `c̃=(1+α)c`、不安全集膨胀 `F_δ={x:d(x,F)≤δ}` | §IV-C | 同上 | 本阶段不实现 |
| 高层 MCRA：reachability critic + budget-conditioned actor + 两阶段最小预算搜索 | §III-A | `CaMRA/navigation/`（仅 README）+ `RC-PPO/` 只读参考 | 本阶段不实现 |
| 两阶段流程：Stage 1 训练冻结+校准 / Stage 2 高层学习+预算选择 | §IV-A | 对应本计划 phase B–D | phase 1 只做 Stage 1 的 tracker |

**分层为何能缓解联合学习困难**（论文 §IV-B）：若直接做关节级 MCRA 学习，稳定行走必须与目标到达、
障碍规避、代价最小化同时学出来；跌倒与过度倾斜虽可编码为不安全状态，但"仅避免这些状态"对
协调、平滑行走的指导有限，且早期频繁失稳会中断导航探索。因此把运动稳定从高层决策中分离。

**冻结为何是校准的前提**：tracker 冻结后执行接口不再随低层参数更新而漂移，
校准数据、高层学习与部署才使用同一个接口，残余失配才能被稳定度量并转成 `α`/`δ`。

**频率差异（仅标注，不在本任务处理）**：论文 §V 实验与真机部分采用高层决策间隔 0.2 s（5 Hz）
与低层 50 Hz（main.tex 第 392、637 行）；本计划改用 200/50/10 Hz。登记与取舍属 TASK-003 / TASK-006。

## 8–9. 待后续任务补充

（TASK-009 补观测/动作/频率契约，TASK-011 补评估与导出要求）

## 10. 论文/历史稿设定 vs 当前代码：差异登记（TASK-003）

### 10.1 数据性质（决定性事实）

`manuscripts/CCDC/data/1029/manuscript_completion_20261026/protocol.json` 自述：

```json
"data_status": "internal_synthetic_working_manuscript"
"empirical_validation": false
```

同目录 `README.md` 原文：**"全部新增配置及结果为 authored_example 或 computed_from_synthetic，不是实测"**，
并明确 **"不能把这些文件改名为 empirical 后当成采集证据"**。
`hardware_results.json` 保留 30 次历史数据，硬件对照已迁至 `data/1029/hardware_comparison_20261029/`（两方法各 50 次）。

### 10.2 事实分级

| 级别 | 来源 | 代表内容 | 能否作为训练达标证据 |
| --- | --- | --- | --- |
| 已验证源码事实 | 本仓库当前代码（逐行核对） | 45 维观测、scale 0.25、动作 clip ±100、decimation 4、kp 25/kd 0.5 | 可作"当前配置"，**不是性能证据** |
| 论文 / 历史设定 | `latex/1020/main.tex` 附录 A + `protocol.json.low_level` | 48 维、`q_des=q_stand+0.25·clip(a,-1,1)`、50/200 Hz、Kp 20、命令域 [0.6,0.4,1.2]、1000 updates | 仅作参考，**不得当当前可执行配置** |
| 合成 / 作者示例数据 | `main_results.json`、`calibration_results.json` 等 | SR 73.3%、BVR 9.0%、α=0.09/δ=0.18、900 trials/方法 | **不可**作为实测或达标证据 |

### 10.3 差异登记（按 implementation_clarifications）

| 项 | 论文 / 历史稿设定 | 当前代码实测 | 计划已定处理 |
| --- | --- | --- | --- |
| 高层频率 | 决策间隔 0.2 s（5 Hz）；一条命令保持 **10 次** tracker 更新 | decimation 4、dt 0.005 → 策略 50 Hz；无高层 | 采用 **200/50/10 Hz**，旧论文不改 |
| 策略观测 | **48 维**（含机身线速度 3 维） | **45 维** = 角速度3 + 重力投影3 + 命令3 + 关节相对位置12 + 关节速度12 + 上一步动作12；**不含 base_lin_vel** | 以 45 维为静态预期，**不为凑 48 维新增传感量** |
| PD 增益 | Kp=20 Nm/rad、Kd=0.5 | stiffness **25.0**、damping 0.5 | 保留 25/0.5 作基线，不混用旧参数 |
| 动作裁剪 | `q_des=q_stand+0.25·clip(a,-1,1)` 再受关节限位 | 动作项 scale 0.25、clip (-100,100)；train/play 另传 `clip_actions` | 必须核全链（策略→wrapper→动作项→位置目标→PD）后再定，**不直接改 ±1** |
| 命令域 | vx∈[-0.6,0.6]、vy∈[-0.4,0.4]、ωz∈[-1.2,1.2] | vx∈[-1.0,1.0]、vy∈[-0.4,0.4]、ωz∈[-1.0,1.0] | **已定（用户 2026-10-31）：保留代码值 ±1.0/±0.4/±1.0**，不采用论文 ±0.6/±1.2；由 TASK-016 落地到配置并核全链 |
| 训练迭代 | 1000 updates | `max_iterations` 默认 50000 | 由实际资源与预算决定（TASK-013/017） |
| 回合与终止 | 20 s；roll/pitch > 0.8 rad 或身高 < 0.18 m 终止 | 未逐项核对 | TASK-009/010 核对 |

### 10.4 不得当作实测结果的项

- 5 Hz / 0.2 s 高层间隔（历史设定，非本项目实测频率）
- 900 trials/方法、800 校准 + 400 held-out 的样本量（合成计数）
- α=0.09 / δ=0.18 / 预算扫描 0.27 等裕度数值
- SR 73.3% / BVR 9.0%、未校准 70.7% / 14.0% 等全部性能数字
- 真机 30 次（旧）与 50 次（新）对照数据
- 历史 PDF 哈希问题：**仅记录外部报告与待追溯状态**，本任务不做整篇论文审计

## 11. 原始最小成本到达避障方法的适用边界（TASK-004）

**身份核对**（`source_manifest.json` 中 `key=rcppo`）：arXiv **2410.22600**，
*Solving Minimum-Cost Reach Avoid using Reinforcement Learning*，27 页；
正文抽取文本 `text/rcppo.txt` 共 1723 行。

### 11.1 问题定义（§3，文本第 150–200 行）

系统为元组 `M = ⟨X, U, f, c, g, h⟩`，状态按**确定性离散动力学**演化：

```
x_{t+1} = f(x_t, u_t)                                    (1)
min_{π,T} Σ_{t=0}^{T-1} c(x_t, π(x_t))                    (3a)
s.t. x_T ∈ G,  x_t ∉ F (∀t≤T),  x_{t+1} = f(x_t, π(x_t)) (3b–3d)
G := {x : g(x) ≤ 0},  F := {x : h(x) > 0}                 (2)
```

两个关键结构特点：**T 本身是决策变量**（灵活终止时刻），且**到达约束只在终端 T 施加**。
原文明确指出正是这两点使常规 RL 方法无法直接套用。

### 11.2 求解思路（§3.2、§4，文本第 208–315 行）

- 把原问题**抬升为增广系统**（额外二元安全状态 + 剩余预算 `z`），在增广目标 `ĝ` 上定义 reach value `Ṽ_ĝ`；
- 能到达增广目标的**最小预算上界 `z0`** 即对应最小代价策略（epigraph 重表述）；
- RC-PPO 分两阶段：**Phase 1** 学 z-conditioned 策略与价值函数；**Phase 2** 做最小预算搜索。

### 11.3 适用边界（原文自述局限，文本第 640–650 行）

1. **安全被折叠进目标**：增广动力学用二元状态记录是否进入不安全集，代价是
   **两个都无法到达目标的策略可能得到相同价值**，即使其中一个是不安全的。
2. **理论依赖确定性动力学假设**，原文原话：该假设
   *"can be quite restrictive as it precludes the use of commonly used techniques for real-world
   deployment such as domain randomization"*，作者将其列为未来工作。

这两条正是 CaMRA 必须分层的根本原因：**低层 tracker 必须用域随机化才能 sim-to-real（需要随机性），
而 RC-PPO 的理论保证建立在确定性假设上（排除随机化）**。把两者合成单一策略，等于让一方的前提作废；
因此改为「冻结低层 + 校准残余失配（α/δ）+ 上层理论适用」的分工。

### 11.4 为什么上层目标不能替代低层的密集学习信号

先明确界限：**不能说理论绝对不允许关节级控制**——MCRA 形式上可以接受任意动作空间。
真正的阻碍是**高维控制的学习难度**与**信号性质**：

| 方面 | 上层 MCRA 信号 | 低层需要的信号 |
| --- | --- | --- |
| 时间结构 | 到达约束**只在终端 T** 施加；T 本身是决策变量 | 每个控制步都要有反馈（50 Hz） |
| 安全表达 | 只有"是否进入 F"的二元指示 | 需要连续的姿态/高度/力矩/平滑度约束 |
| 代价 | 累积导航代价（与平衡、步态无关） | 需要跟踪误差、foot airtime、动作平滑等整形项 |
| 价值区分度 | 见 11.3 第 1 条：不安全但未到达的策略可能同值 | 需要能区分"稳但慢"与"快但会摔" |

因此「避免跌倒」虽然能在 MCRA 里编码为不安全状态，但**仅避免这些状态对协调、平滑行走的指导很有限**，
且早期频繁失稳会中断高层探索（本项目 1029 论文 §IV-B 亦有同样论述）。

本任务只做概念整理，不展开定理证明，也不重新检索无关文献。

## 12. RC-PPO 参考代码调用链（TASK-005）

> 副本冻结：本任务**不安装其 JAX 环境、不修复导入、不移植上层算法**，只读源码。

### 12.1 调用链

```
script/run_<exp>.sh
  └─ rl/arguments.py : get_args()            读实验参数
  └─ env/env_list.py : get_env(config)       按 config["EXP_NAME"] 分支构造环境
       └─ env/reach_avoid/<env>.py           具体环境（pendulum / hopper / half_cheetah / wind_field / F16）
       └─ env/wrappers.py : TransformObservation  观测逐维仿射归一
  └─ rl/EC-EFPPO.py : train(env, env_params, config, rng)     主训练循环
       ├─ EFPPO_utils._env_step               采样一步
       ├─ rl/gae.py  calculate_gae / calculate_gae_reach*     双评价各自的 GAE
       ├─ EFPPO_utils._ecefppo_update          策略/价值更新
       └─ flax checkpoints.save_checkpoint     写检查点（路径硬编码，见 12.4）
  └─ rl/root_finding.py : Bisection           预算上界的二分搜索（Phase 2）
```

### 12.2 单个环境的观测/动作/预算/终止语义

以 `env/reach_avoid/hopper_avoid_ceiling.py`（216 行）为准：

| 项 | 实现 |
| --- | --- |
| 状态载体 | `EnvState(state, energy, reach_value, avoid_value)`，`energy` 即**剩余预算**，范围 `[-400.0, 800.0]` |
| 观测 | `obs = concat(env_obs, [avoid_value, energy])`——**末两位固定是“安全指示 + 剩余预算”** |
| 预算初始化 | `reset` 中 `init_energy ~ U(min_energy, max_energy)` |
| 代价 | `step` 只有在**超出力矩上限**时才计 `energy_consumption = (|u·gear/2|²)·0.6`，逐关节求和 |
| 预算更新 | `next_energy = clip(energy − consumption, min_energy, max_energy)` |
| 返回值 | `(obs, next_state, energy_consumption, next_state.done > 0.5, pos_dict)` |
| 终止 | 由底层环境自身的 `done` 决定（如高度/角度越界），**不是**上层预算耗尽 |

要点：**预算耗尽是"代价"而非"终止"**，安全进入由 `avoid_value` 记录；这与本项目要求的
"预算为负不早停、安全进入置持久标志"一致。

### 12.3 网络结构与联合评价（model/actorcritic.py，323 行）

| 组件 | 实现 |
| --- | --- |
| `Policy_Network` | 2×256 `tanh` → `action_dim`；`log_std` 是**与状态无关的可学习参数**；输出 `distrax.MultivariateNormalDiag` |
| `Value_Network` | 独立 MLP，供 reach / cost 两路评价 |
| `IQE` / `iqe()` | 区间并集的期望（对多个区间取并后积分），用于对 `max` 求期望 |
| `MaxMean` | 带可学习 `raw_alpha`（sigmoid）的 max/mean 混合 |

**联合评价与最小化符号**（`rl/EC-EFPPO.py` 第 65–66 行）：

```python
V_total = jnp.maximum(V_reach_append, V_append - energy_append)   # max(V_reach, V_cost − z)
g_append = jnp.maximum(reach_append,  -energy_append)             # max(g, −z)
```

与论文 `ĝ = max{g, Cy, −z}` 完全对应（`energy` 即 `z`）。**优化方向是最小化 `V_total`（越负越好）**，
即"先满足到达与安全，再压低代价"的字典序式耦合。

**与普通 PPO 的区别**：状态含预算 z、双价值网络（reach + cost）经 max 耦合、存在 stop/终止选择、
Phase 2 用二分搜索找最小可行预算（`root_finding.Bisection`）。

### 12.4 不能直接运行的原因（源码依据）

| 原因 | 证据 |
| --- | --- |
| 包路径不存在 | `rl/EC-EFPPO.py:2` `sys.path.append("/home/mepear_gc")`，随后 `from EFPPO.src.rl.EFPPO_utils import ...`——当前布局已无 `src/` 且无 `EFPPO` 包 |
| 检查点路径硬编码 | `rl/EC-EFPPO.py:198` `ckpt_dir='/home/mepear_gc/EFPPO/model/{}/'` |
| 依赖缺失 | 头部导入 `wandb / jax / flax / distrax`，环境侧依赖 `mujoco`；本机未安装（本任务不安装） |
| 脚本与环境不匹配 | `script/` 只有 5 个 `run_*.sh`，README 提到的 `run_f16_avoid.sh` **缺失**；`run_hopper_reach.sh` 的 `HopperReach` 未在 `get_env` 注册；`arguments.py` 默认的 `GridConstraint` 亦不在分支内 |
| 调试残留 | 头部 `import pdb` |
| 作者自述 | `README.md` 第 71 行：*"未执行训练或验证实验结果…不能当作已经配置完成的可运行发行版"* |

### 12.5 未来上层对底层接口的要求（提取）

1. 底层**只接收** 3 维机身速度指令 + 本体状态，输出 12 关节命令；
   **不接收**预算 z、安全位 y、reach/avoid 值——这些由上层维护。
2. 固定频率：底层 50 Hz 推理，上层 10 Hz 下发并保持（旧稿为 5 Hz 保持 10 次 tracker 更新）。
3. 命令定义在**机身坐标系**，单位 m/s 与 rad/s，需限幅（本项目已定 ±1.0/±0.4/±1.0）。
4. 上层需要每个决策步可获得的**实际位移/速度**以计算 cost，以及**可复现的复位接口**（同初值、同地图、同命令序列）。
5. 代价由上层按导航代理成本计算（非关节力矩），因此**底层不产生 cost 信号**。

**结论：底层训练不照搬 `max(V_reach, V_cost − z)` 这一最小化目标。**
底层用常规 PPO 的密集运动奖励（速度跟踪 + 姿态 + 力矩 + 动作平滑 + foot airtime），
上层的最小成本目标只在 `navigation` 阶段使用。

## 13. HiRA 分层接口与频率来源（TASK-006）

> HiRA 为只读子模块（`6a05af5`），本任务不修改其代码。README/注释与代码冲突时以代码为准。

### 13.1 频率链（代码验证，非注释推断）

| 参数 | 位置 | 值 | 推导 |
| --- | --- | --- | --- |
| `dt = 0.005` | `base/legged_robot_config.py:155` | 0.005 s | **仿真 200 Hz** |
| `decimation = 4` | `base/legged_robot_config.py:68`、`go2/go2_config.py:123` | 4 | 策略 **50 Hz** = 200/4 |
| `high_level_action_repeat = 5` | `go2/go2_config.py:199` | 5 | 高层 **10 Hz** = 50/5 |

结论：**200 / 50 / 10 Hz** 与计划要求一致，且来自配置继承与覆盖（`GO2HighLevelCfg.env` 继承
`GO2RoughCfg.env` 后新增 `high_level_action_repeat`），不是文档宣称值。

### 13.2 高层→低层执行（`hierarchical_go2_env.py:125–176`）

```python
self.high_level_env.update_velocity_commands(high_level_actions)   # ① 高层动作 → 速度命令
desired_velocity_commands = self.base_env.commands[:, :3].clone()  # ② 命令快照
for _ in range(self.low_level_action_repeat):                      # ③ 重复 5 个低层步
    self.base_env.commands[:, :3] = desired_velocity_commands      #    ★ 每步重写命令
    self.base_env.compute_observations()                           #    ★ 每步重算低层观测
    current_base_obs = self.base_env.get_observations()
    low_level_actions = self.low_level_policy(current_base_obs)    #    ★ 每步重新推理
    base_obs, ..., step_dones, ... = self.base_env.step(low_level_actions)
    aggregated_dones |= step_dones.bool()                          #    ★ 终止按 OR 传播
self.high_level_env._compute_high_level_observations()             # ④ 5 步后才更新高层观测
g_values, h_values = self.high_level_env.compute_g_h_values(...)   # ⑤ 再算 g/h
```

**验收要求标注的四个关键位置**：

| 关键点 | 代码位置 | 原因 |
| --- | --- | --- |
| 命令必须每步重写 | `for` 循环内第 149 行赋值 | 低层环境的随机命令课程会覆盖高层指令，不重写则 5 步内命令漂移 |
| 观测必须每步重算 | 循环内 `compute_observations()` | 低层策略需要新鲜本体观测，不能用 5 步前的缓存 |
| 策略必须每步推理 | 循环内 `low_level_policy(...)` | 低层 50 Hz，不能在高层 10 Hz 上复用同一动作 |
| 终止按 OR 聚合 | `aggregated_dones |= step_dones` | 5 步内任一子步失败都应结束高层回合 |

### 13.3 动作缩放与命令域（重要差异）

`high_level_navigation_env.py:93–107`：

```python
high_level_actions = torch.clip(high_level_actions, -1.0, 1.0)
velocity_commands  = high_level_actions * self.action_scale      # action_scale 默认 [1.0,1.0,1.0]
self.base_env.commands[:, 0] = velocity_commands[:, 0] * 0.6    # vx
self.base_env.commands[:, 1] = velocity_commands[:, 1] * 0.2    # vy
self.base_env.commands[:, 2] = velocity_commands[:, 2] * 0.8    # vyaw
```

- 有效命令域 = **vx ±0.6、vy ±0.2、vyaw ±0.8**
- 与 1020 历史稿（0.6 / 0.4 / 1.2）不同，与本项目代码（1.0 / 0.4 / 1.0）也不同
- `0.6 / 0.2 / 0.8` 是**硬编码常量**，没走 `action_scale` 机制；若复用必须参数化，否则会与
  TASK-016 命令域冲突

### 13.4 复位链（已核实无缺口）

`hierarchical.reset()` → `high_level.reset()` → **第 81 行 `base_env.reset()`** → 再计算高层观测与 g/h。
即一次高层复位会**连带复位低层**，无需额外调用。

### 13.5 与 CaMRA 的评价结构差异

| 方面 | HiRA 高层 | CaMRA 上层 |
| --- | --- | --- |
| 评价结构 | **单评价** Reach-Avoid（一个 RA 值网络 + g/h 标量） | **预算双评价**（reach critic + cost critic） |
| 耦合方式 | 直接用 RA 值做代价 | `max(V_reach, V_cost − z)` |
| 预算 | 无显式预算状态 | 预算 z 入观测并逐步扣减 |

**本阶段仅借鉴衔接方式**（命令写入、重复执行、逐步重算、终止 OR 传播、g/h 计算时机），
**不照搬其单评价结构**。

另注：`go2_config.py` 的 `resume_path` 与 `low_level_model_path` 指向具体日志目录
（`logs/high_level_go2/成功率收敛0.72_成本收敛79.1/model_1300.pt`、
`logs/rough_go2/Sep08_11-57-26_/model_18500.pt`），复用时必须替换。

## 14. HiRA 既有低层策略、观测与部署经验（TASK-007）

> 未执行任何检查点反序列化，未复制任何权重。HiRA 的 `model_18500.pt` 仅作为**参考**，
> 不得当作本项目已完成的 Go2 策略。

### 14.1 低层规格：训练 vs 部署

| 项 | 训练（Isaac Gym / legged_gym） | 部署（`deploy_mujoco/configs/go2.yaml`） |
| --- | --- | --- |
| 观测维度 | 45（`go2_config.py:11`） | 45（`num_obs`） |
| 动作维度 | 12 | 12 |
| 仿真步长 | `dt=0.005` → **200 Hz** | `simulation_dt=0.002` → **500 Hz** |
| 控制抽取 | `decimation=4` → 策略 **50 Hz** | `control_decimation=10` → 策略 **50 Hz** |
| PD 增益 | `stiffness=20`、`damping=0.5` | `kps` 全 20、`kds` 全 0.5 |
| 动作缩放 | `action_scale=0.25` | `action_scale=0.25` |
| 观测缩放 | — | `ang_vel 0.25`、`dof_pos 1.0`、`dof_vel 0.05` |
| 命令缩放 | — | `cmd_scale [2.0, 2.0, 0.25]`、`cmd_init [-0.8, 0, 0]` |

**关键区分**：训练与部署的**仿真步长不同（0.005 vs 0.002），但策略频率都是 50 Hz**。
因此"50 Hz 策略"是可迁移的接口常量，`dt` 不是。

### 14.2 低层观测排列（`go2/go2_env.py:31–41`）

```
[ base_ang_vel(3) , projected_gravity(3) , commands(3) ,
  (dof_pos − default_dof_pos)(12) , dof_vel(12) , actions(12) ]  = 45
```

与当前 `unitree_rl_lab` 的策略观测**块顺序完全一致**，维度同为 45。

### 14.3 兼容项 / 不兼容项清单

| 项 | HiRA | 当前 `unitree_rl_lab` | 结论 |
| --- | --- | --- | --- |
| 观测块顺序 | 角速度→重力→命令→关节位置→关节速度→动作 | 同 | ✅ **兼容** |
| 观测维度 | 45 | 45 | ✅ 兼容 |
| 动作维度 / 缩放 | 12 / 0.25 | 12 / 0.25 | ✅ 兼容 |
| **关节排列** | **FL, FR, RL, RR** | **FR, FL, RR, RL** | ❌ **不兼容**：12 维块内顺序必须重排 |
| 默认姿态数值 | hip ±0.1；前腿 thigh 0.8；后腿 1.0；calf −1.5 | 同值（hip R−0.1/L+0.1、F 0.8、R 1.0、calf −1.5） | ✅ 数值一致，仅随排列变化 |
| PD 增益 | 20 / 0.5 | **25** / 0.5 | ⚠️ 不一致（本项目保留 25，见 §10.3） |
| **角速度观测缩放** | **0.25** | **0.2** | ❌ 不兼容：权重不能直接迁移 |
| 关节速度缩放 | 0.05 | 0.05 | ✅ 兼容 |
| 命令缩放（部署侧） | `cmd_scale [2,2,0.25]` | 命令直接写入，无额外缩放 | ❌ 部署侧需适配 |
| 仿真步长 | 训练 0.005 / 部署 0.002 | 0.005 | ⚠️ 训练一致、部署不同 |

关节排列具体顺序（源码）：

```
HiRA deploy go2.yaml : FL_hip, FL_thigh, FL_calf, FR_hip, ... , RL_hip, ... , RR_hip, ...
unitree_rl_lab       : FR_hip, FR_thigh, FR_calf, FL_hip, ... , RR_hip, ... , RL_hip, ...
```

### 14.4 可复用的经验

1. **接口节拍**：策略 50 Hz、高层 10 Hz（TASK-006 已由代码确证）。
2. **观测块顺序**：45 维六段结构与新框架一致，无需重新设计。
3. **动作语义**：`q_des = q_default + 0.25·a`，与新版 `JointPositionAction(scale=0.25, use_default_offset=True)` 同构。
4. **部署侧数值**：12 关节 kp/kd 与默认姿态角可直接沿用作为 MuJoCo 参考。

### 14.5 不能直接迁移的假设

1. **权重本身不可迁移**：关节排列（FL-first vs FR-first）与角速度缩放（0.25 vs 0.2）都不同；
   HiRA 的 `model_18500.pt` 直接加载到新框架会得到错误行为。且按任务约束，参考权重不得当作本项目训练成果。
2. **训练栈不同**：HiRA 是 Isaac Gym + legged_gym + rsl_rl；README 第 34 行明确
   *"Renaming the local folder does not migrate the implementation to Isaac Lab or change the trained policies"*。
3. **部署命令缩放冲突**：`cmd_scale [2,2,0.25]` 会放大约 2 倍线速度，与本项目已定的
   ±1.0/±0.4/±1.0 命令域不一致，部署侧需重写映射。

## 15. 宇树框架的注册、启动与资产链（TASK-008）

### 15.1 启动链

```
unitree_rl_lab.sh            要求已激活 conda（取 ${CONDA_PREFIX}/bin/python，未激活时报 Error）
  └─ scripts/rsl_rl/train.py
       ├─ @hydra_task_config(task, "rsl_rl_cfg_entry_point")
       │     └─ 从 gym registry 的 kwargs 取出 env_cfg 与 agent_cfg
       ├─ CLI 覆盖：num_envs / max_iterations / seed / sim.device / distributed
       ├─ 日志目录：logs/rsl_rl/{experiment_name}/{YYYY-MM-DD_HH-MM-SS}_{run_name}
       ├─ gym.make(task, cfg=env_cfg)            → ManagerBasedRLEnv
       ├─ RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
       ├─ OnPolicyRunner(env, agent_cfg.to_dict(), log_dir, device)
       └─ 落盘：params/env.yaml、params/agent.yaml、export_deploy_cfg(...)、env_cfg 源文件副本
```

`export_deploy_cfg(...)` 与 `params/` 目录是后续 TASK-011（导出与部署接入）的现成入口。

### 15.2 注册链（三件套绑定）

```
tasks/__init__.py        → import_packages(__name__, _BLACKLIST_PKGS)   # 递归导入子包
locomotion/__init__.py   → from .robots import *
robots/go2/__init__.py   → gym.register(...)
```

```python
gym.register(
    id="Unitree-Go2-Velocity",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point":      f"{__name__}.velocity_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point":   "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)
```

要点：**任务 ID → 环境配置类 → 训练/评估配置类** 通过 gym registry 一次绑定；
`hydra_task_config` 靠这个名字取配置，所以**改任务 ID 必须同步改脚本参数**。

### 15.3 资产链与缺口

```
unitree.py:  UNITREE_MODEL_DIR  →  usd_path = f"{UNITREE_MODEL_DIR}/Go2/usd/go2.usd"
```

| 环节 | 内容 |
| --- | --- |
| 上游声明 | 从 HuggingFace 数据集 `unitreerobotics/unitree_model` 克隆并**保持目录结构**，再配置 `UNITREE_MODEL_DIR`（`README.upstream.md` 第 40–50 行） |
| 本项目现状 | 已指向 `/pub/data/caohy/unitree_model`，4 个 USD 文件齐备，冒烟阶段已验证可打开（顶层节点 `/go2_description`） |
| 取源差异 | 本次取自同名 **GitHub** 仓库的 LFS 媒体地址（该仓库 `.usd` 走 Git LFS），路径结构 `Go2/usd/...` 与上游一致；如需严格遵循上游可改用 HF 克隆 |

### 15.4 上游声明 vs 本项目已验证状态

| 项 | 上游声明 | 本项目状态 |
| --- | --- | --- |
| Isaac Lab / Isaac Sim | 2.3.0 / 5.1.0（README 徽章） | ✅ Tiger12 实装并冒烟通过 |
| 任务注册 | `Unitree-Go2-Velocity` | ✅ `list_envs.py` 输出可见 |
| Go2 USD 资产 | 需下载并配置 `UNITREE_MODEL_DIR` | ✅ 4 文件就位，USD 可打开 |
| rsl-rl 版本 | 2.3.0 钉 `rsl-rl-lib==3.0.1` | ✅ 已装 3.0.1 |
| 正式训练 | 提供入口 | ⚠️ **尚未跑过** |
| C++ 部署（`deploy/`） | 提供 CMake 工程 | ⚠️ 未构建、未验证 |

### 15.5 可配置设备参数 vs 影响控制语义的配置

**可自由调整**（只影响资源与产出位置，不改变控制语义）：
`num_envs`、`max_iterations`、`sim.device`、`seed`、日志目录名、`--video` 相关参数。

**不可随意调整**（会改变控制语义或与已定决策冲突）：
`sim.dt`、`decimation`、命令域范围、动作 `scale`/`clip`、PD 增益、观测缩放与排列、关节顺序。

**目标设备需验证的依赖**（Tiger12 已完成，此处存档要求）：驱动与 Vulkan、USD 可读性、
`rsl-rl-lib` 版本、torch/numpy 钉版一致性、`/pub` 空间。

## 16. Go2 观测、动作与物理控制契约（TASK-009）

### 16.1 Actor 观测逐字段（45 维，静态推导）

来源 `go2/velocity_env_cfg.py` 的 `PolicyCfg`，**字段顺序即拼接顺序**（`concatenate_terms = True`）：

| # | 字段 | 维度 | 缩放 | 裁剪 | 噪声 (均匀) | 坐标系 / 语义 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | `base_ang_vel` | 3 | **0.2** | ±100 | ±0.2 | 机身系角速度 rad/s |
| 2 | `projected_gravity` | 3 | 1.0 | ±100 | ±0.05 | 机身系重力投影（姿态） |
| 3 | `velocity_commands` | 3 | 1.0 | ±100 | 无 | `base_velocity` 命令 [vx,vy,ωz] |
| 4 | `joint_pos_rel` | 12 | 1.0 | ±100 | ±0.01 | 关节相对默认位置 rad |
| 5 | `joint_vel_rel` | 12 | **0.05** | ±100 | ±1.5 | 关节速度 rad/s |
| 6 | `last_action` | 12 | 1.0 | ±100 | 无 | 上一步动作 |
| | **合计** | **45** | | | | |

- `enable_corruption = True`：噪声**仅在训练时**注入，评估/导出需关闭。
- **Critic 特权信息**（60 维）：在 actor 45 维基础上，额外 `base_lin_vel`(3) 与 `joint_effort`(12)，共 3+3+3+3+12+12+12+12 = **60**。
- 约束落实：**actor 不含任何仅仿真可得的量**（无 `base_lin_vel`、无 `joint_effort`），因此部署时 actor 可独立运行。

### 16.2 动作 → 关节控制全链

```
策略输出 a (12 维，高斯，理论无界)
  │
  ├─① RslRlVecEnvWrapper(clip_actions=agent_cfg.clip_actions)
  │     agent_cfg.clip_actions 默认 = None  (rl_cfg.py:178)
  │     → 若为 None 则 **不裁剪**（vecenv_wrapper.py:154 只在非 None 时 clamp）
  │
  ├─② JointPositionAction.process_actions (joint_actions.py:166–175)
  │     processed = raw * scale(0.25) + offset(默认关节角)
  │     再 clamp(processed, cfg.clip = (-100, +100))   ← 作用在**位置目标 rad** 上
  │
  ├─③ 位置目标写入执行器
  │     PD：kp = 25.0、kd = 0.5（UNITREE_GO2_CFG 的 GO2HV）
  │     力矩-速度包络：UnitreeActuatorCfg_Go2HV(X1=13.5, X2=30, Y1=20.2, Y2=23.4)
  │
  └─④ 关节限位（USD 资产内定义的 joint limits）
```

**关键结论：`[-100, 100]` 不是有效动作范围。** 它施加在 `raw*0.25 + default` 之后的位置目标上（单位 rad），
当 raw 动作在常规量级（|a| ≤ 数个单位）时该裁剪**永不触发**；真正约束机器人行为的是
**关节位置限位 + 力矩-速度包络 + PD 增益**。计划中「不要直接改成 [-1,1]」的警告即针对此点。

因此若上层需要「严格有界的动作」，正确做法是显式设置 `agent_cfg.clip_actions`（对 raw 动作裁剪），
而不是依赖动作项的 ±100。

### 16.3 控制频率

| 层 | 频率 | 依据 |
| --- | --- | --- |
| 物理仿真 | **200 Hz**（dt = 0.005） | `SimulationCfg.dt` |
| 策略推理 | **50 Hz**（decimation = 4） | `self.decimation = 4`，200/4 |
| 渲染 | 与策略同步 | `sim.render_interval = decimation` |

即仿真每 4 步跑一次策略；`last_action` 观测项在 decimation 周期内保持不变。

### 16.4 SDK 与仿真关节顺序

`UNITREE_GO2_CFG.joint_sdk_names` 的顺序为：

```
FR_hip, FR_thigh, FR_calf, FL_hip, FL_thigh, FL_calf,
RR_hip, RR_thigh, RR_calf, RL_hip, RL_thigh, RL_calf
```

即 **FR → FL → RR → RL**。该顺序与 HiRA（FL → FR → RL → RR）**不同**（见 §14.3），
部署到实机或复用 HiRA 经验时必须重排。

> 补充（TASK-015 运行时实测）：**仿真侧 `joint_names` 顺序又不同**——它是按关节类型分组
> `FL_hip, FR_hip, RL_hip, RR_hip, FL_thigh, ...`。因此仿真→SDK 的重排必须依赖
> `joint_ids_map`，详见 §22.4。

### 16.5 基线确认

- 观测维度基线：**45 维**（静态推导）。
- PD 基线：**kp = 25.0 / kd = 0.5**（不采用 HiRA 与 1020 稿的 kp=20）。
- 以上两项均标记为**静态推导**，须在 TASK-015 / TASK-021 用运行时张量复核实际维度与顺序。

## 17. 底层训练目标、课程与扰动机制（TASK-010）

### 17.1 奖励结构

| 类型 | 项 | 权重 | 含义 |
| --- | --- | --- | --- |
| 跟踪（正） | `track_lin_vel_xy_exp` | **+1.5** | `exp(-‖v_xy−cmd_xy‖²/0.25)`，std = √0.25 = 0.5 |
| 跟踪（正） | `track_ang_vel_z_exp` | **+0.75** | 同上形式，偏航角速度 |
| 正则（负） | `lin_vel_z_l2` | −2.0 | 抑制竖直速度 |
| 正则（负） | `ang_vel_xy_l2` | −0.05 | 抑制横滚/俯仰角速度 |
| 正则（负） | `joint_vel_l2` | −0.001 | 关节速度 |
| 正则（负） | `joint_acc_l2` | −2.5e−7 | 关节加速度 |
| 正则（负） | `joint_torques_l2` | −2e−4 | 力矩 |
| 正则（负） | `action_rate_l2` | −0.1 | 动作变化率（平滑） |
| 正则（负） | `joint_pos_limits` | −10.0 | 关节限位越界 |
| 正则（负） | `energy` | −2e−5 | 能耗 |
| 正则（负） | `flat_orientation_l2` | −2.5 | 保持机身水平 |
| 正则（负） | `joint_position_penalty` | −0.7 | 偏离默认姿态 |
| 步态（正） | `feet_air_time` | +0.1 | 抬足时间（鼓励迈步） |
| 步态（负） | `air_time_variance_penalty` | −1.0 | 四足抬足时间方差 |
| 步态（负） | `feet_slide` | −0.1 | 支撑相打滑 |
| 接触（负） | `undesired_contacts` | −1.0 | 非足部接触 |

与 1020 稿相比：结构同源（跟踪 + 竖直速度 + 姿态 + 力矩 + 平滑 + 抬足），
但**新增** `joint_vel`、`joint_acc`、`energy`、`flat_orientation`、`feet_slide`、`air_time_variance`，权重亦不同。

### 17.2 命令课程（关键）

```python
base_velocity = mdp.UniformLevelVelocityCommandCfg(
    resampling_time_range=(10.0, 10.0),
    rel_standing_envs=0.1,
    ranges       = Ranges(lin_vel_x=(-0.1, 0.1), lin_vel_y=(-0.1, 0.1), ang_vel_z=(-1, 1)),      # 初始
    limit_ranges = Ranges(lin_vel_x=(-1.0, 1.0), lin_vel_y=(-0.4, 0.4), ang_vel_z=(-1.0, 1.0)),  # 上限
)
```

- **命令域是课程逐步放大的**：从 `±0.1 / ±0.1 / ±1.0` 起步，由课程项 `lin_vel_cmd_levels`
  逐步扩到 `limit_ranges` 的 `±1.0 / ±0.4 / ±1.0`。
- `rel_standing_envs = 0.1`：10% 环境命令置零，专门练习站立。
- 每 10 s 重采样一次命令。

**意义**：用户决策保留的 `±1.0/±0.4/±1.0` 是**上限**，不是训练起始值；
直接全范围起步会显著加大早期学习难度。

### 17.3 扰动与域随机化

| 事件 | 模式 | 参数 |
| --- | --- | --- |
| `randomize_rigid_body_material` | startup | static/dynamic friction **(0.3, 1.2)**、restitution (0.0, 0.15) |
| `randomize_rigid_body_mass` | startup | 附加质量 **(−1.0, 3.0)** kg |
| `apply_external_force_torque` | reset | `force_range (0.0, 0.0)` → **实际关闭** |
| `reset_root_state_uniform` | reset | 初速度随机 |
| `reset_joints_by_scale` | reset | 关节速度 (−1.0, 1.0) |
| `push_by_setting_velocity` | interval | **每 5–10 s**，x/y 速度冲击 **(−0.5, 0.5)** m/s |

与 1020 稿对比：

| 项 | 1020 稿 | 当前代码 | 判断 |
| --- | --- | --- | --- |
| 摩擦 | [0.5, 1.25] | **[0.3, 1.2]** | 更宽（更难） |
| 质量 | 倍数 ×[0.9, 1.1] | **附加 −1~+3 kg** | 更宽（更难） |
| 推力 | 0.25 m/s / 8 s | **0.5 m/s / 5–10 s** | 更强（更难） |
| 命令延迟 | 0–40 ms | **无** | ⚠️ **新版缺失** |
| 观测噪声 | 0.02 m/s、0.03 rad/s | 见 §16.1 各 obs 噪声 | 形式不同，需在 TASK-021 实测 |

命令延迟随机化缺失是**已知缺口**，属 TASK-019 的可配置项，本阶段不擅自添加。

### 17.4 终止条件

| 项 | 判据 |
| --- | --- |
| `time_out` | `episode_length_s = 20.0` |
| `base_contact` | base 接触力 > 1.0 → 失败 |
| `bad_orientation` | 倾角 > **0.8 rad** |

对比 1020 稿：同样 20 s、0.8 rad；但**新版没有「身高 < 0.18 m」终止项**，属差异，记录待核。

### 17.5 地形课程

`CurriculumCfg`：`terrain_levels_vel`（地形难度）与 `lin_vel_cmd_levels`（命令范围）。
地形 `COBBLESTONE_ROAD_CFG`，`num_rows = 10`，`flat` 子地形占比 0.1。

## 18. 评估、导出与未来模块接入要求（TASK-011）

### 18.1 评估入口 `scripts/rsl_rl/play.py`

| 环节 | 实现 |
| --- | --- |
| 检查点定位 | `get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)` |
| 取网络 | `try: runner.alg.policy / except: runner.alg.actor_critic`（rsl-rl ≥ 2.3 走前者；3.0.1 满足） |
| 归一化器 | `actor_obs_normalizer` → `student_obs_normalizer` → `None` 依次尝试 |
| 导出目录 | **`os.path.dirname(resume_path)/exported/`**（检查点同级） |
| 导出文件 | `policy.pt`（JIT）与 `policy.onnx`，**均携带 normalizer** |
| 回放节拍 | `dt = env.unwrapped.step_dt`（0.005×4 = **0.02 s = 50 Hz**），`--real_time` 时按此 sleep |
| 版本兼容 | `if version("rsl-rl-lib").startswith("2.3.")` 分支处理 `get_observations()` 返回值差异 |

**重要缺口**：`play.py` **只回放，不做指标计算**——没有速度误差、停车漂移、失败率等任何统计。
评估工具必须自建（TASK-020），不能把"能跑起来"当作评估。

### 18.2 冻结包内容（`export_deploy_cfg`）

训练结束时 `train.py` 调用 `export_deploy_cfg(env, log_dir)`，`utils/export_deploy_cfg.py` 写出：

| 字段 | 来源 | 用途 |
| --- | --- | --- |
| `joint_ids_map` | 仿真关节 → `joint_sdk_names` 的匹配映射 | 部署侧关节重排（关键防错位） |
| `step_dt` | `sim.dt × decimation` = 0.02 | 控制周期 |
| `stiffness` / `damping` | 按 `joint_ids_map` **重排后**的每关节 PD | 部署 PD |
| `default_joint_pos` | 资产默认姿态 | 动作偏置 |
| `commands.base_velocity.ranges` | **优先取 `limit_ranges`**，无则取 `ranges` | 命令限幅（→ 自动为 ±1.0/±0.4/±1.0） |
| `actions`（含 `scale`） | 动作项配置，标量会展开成 12 维列表 | 动作映射 |

这一份配置 + `policy.pt/.onnx` + 资产版本，就是 calibration 所需"可重载的冻结控制器"。

### 18.3 部署侧接口（`deploy/robots/go2/config/config.yaml`）

| 状态机 | id | 内容 |
| --- | --- | --- |
| `Passive` | 1 | kd = 3（全关节阻尼） |
| `FixStand` | 2 | kp = [60,80,80]×4、kd = [5,4,4]×4，qs 三段插值到站立姿态 |
| `Velocity` | 3 | `type: RLBase`，`policy_dir: ../../../logs/rsl_rl/unitree_go2_velocity` |

**注意**：`FixStand` 的 kp 60/80/80 与 RL 策略的 **25/0.5 完全不同**（站立用高刚度），
配置时不可混用；这两套增益服务于不同状态机阶段。

### 18.4 未来模块的接入要求

**对 calibration 模块**：

1. 冻结控制器三件套：`policy.pt/.onnx` + `export_deploy_cfg` 配置 + USD/资产版本与散列。
2. **可复现复位接口**：同初值、同地图、同命令序列，才能做名义/执行配对 rollout。
3. **逐步记录**：机体位置、速度、命令、时间戳（配对时按共同 0.2 s 网格线性插值对齐）。
4. **名义轨迹生成器**：α/δ 需要 `J_nom`，即"理想跟踪下积分命令速度"的抽象闭环（本项目尚未实现）。

**对 navigation 模块**：

1. 只接收/下发 3 维机身速度命令，10 Hz 下发、50 Hz 执行（§13.1）。
2. 不得让底层消费预算 z 或安全位 y（§12.5）。
3. 命令限幅必须与训练 `limit_ranges` 一致：**±1.0 / ±0.4 / ±1.0**（用户已定）。

### 18.5 导出前必须核对的数值一致性（对应 TASK-025）

| 检查项 | 判据 |
| --- | --- |
| `step_dt` 一致 | 导出配置的 `step_dt` 与 `play.py` 的 `env.unwrapped.step_dt` 相等（0.02） |
| normalizer 随模型 | `policy.pt/.onnx` 内嵌 normalizer，否则观测缩放不一致 |
| `joint_ids_map` 正确 | 覆盖 12 关节、无重复、顺序与 `joint_sdk_names` 一致 |
| 动作 scale | 导出为 12 维列表且每项 0.25 |
| 命令域 | 导出为 `limit_ranges` 值 ±1.0/±0.4/±1.0 |

## 19. 学习结论汇总与本阶段实施边界（TASK-012）

### 19.1 五方对应表

| 论文要求 | RC-PPO 参考实现 | HiRA 经验 | 宇树实现（新框架） | 本次决定 / 改动 |
| --- | --- | --- | --- | --- |
| 分层：冻结速度接口 | **无分层**，单策略直接输出动作 | **有分层**：高层 10 Hz + 冻结低层 50 Hz | 低层 velocity tracker | 采用宇树低层；上层留待 navigation |
| 机身三维速度命令 | 抽象动作 `u_t`，无关节语义 | 3 维 `[vx,vy,vyaw]`，clip±1 后**再乘 0.6/0.2/0.8**（硬编码） | 命令域 **±1.0/±0.4/±1.0**（课程上限） | **已定：保留 ±1.0/±0.4/±1.0** |
| 12 关节目标 | — | `q_des=q_def+0.25a`，kp **20** | `q_def+0.25a`，kp **25** / kd 0.5 | **保留 25/0.5** |
| α/δ 执行失配校准 | — | 无（单评价 RA） | — | **本阶段不做**，留 calibration |
| 导航成本 / footprint | 导航代理成本（合成数据） | — | — | **本阶段只保存资产几何与版本** |

### 19.2 已固定的实施边界

**本阶段要做**：

1. 只做 **Go2 平地速度跟踪**；地形课程沿用现成配置，不改其语义。
2. 节拍：**仿真 200 Hz / 策略 50 Hz / 外部命令 10 Hz**。
3. 策略契约：actor **45 维**、critic **60 维**、**12 关节**动作、`scale=0.25`、`kp=25/kd=0.5`。
4. 命令域：**±1.0 / ±0.4 / ±1.0**（课程从 ±0.1 起步逐步放大）。
5. 训练在 **Tiger12（L40）**、`tmux` 内执行、检查点与日志落 `/pub`。
6. 交付冻结包：`policy.pt/.onnx` + `export_deploy_cfg` 配置 + 资产版本与散列。

**本阶段不做**：

1. calibration（α/δ）与 navigation（MCRA 上层）。
2. 真机上电、MuJoCo 完整迁移、感知与障碍规避训练。
3. 修改论文、RC-PPO、HiRA 任何代码。
4. 把 HiRA 权重或论文合成数据当作本项目结果。
5. 改动命令域与已钉版本之外的核心依赖。

### 19.3 必须在设备阶段补齐的事实

| 事实 | 负责任务 | 现状 |
| --- | --- | --- |
| Tiger12 实测吞吐（steps/s）与显存占用 | TASK-013 / TASK-022 | **未测** |
| 45 维观测运行时张量与顺序 | TASK-015 | 静态推导 |
| 动作全链实际有效范围 | TASK-015 / TASK-021 | 静态推导 |
| 训练预算（种子数、时长） | TASK-017 + 用户 | **未定** |
| 评估阈值（速度误差/响应/停车漂移/失败率） | TASK-017 + 用户 | **为空** |
| Go2 USD 版本散列 | TASK-014 | 已下载未散列 |

### 19.4 实质性未决项（需用户决策）

1. **评估阈值具体数值** —— 计划要求训练前固定，不能事后拿论文数字凑（论文数字为合成数据）。
2. **训练种子数与总时长预算**。
3. 是否补回**命令延迟随机化（0–40 ms）**与**身高终止（<0.18 m）** —— 当前新版缺失。
4. **4096 环境是否需要下调** —— L40 仅 46G 显存，需实测。

已决项：命令域 ±1.0/±0.4/±1.0（用户 2026-10-31 决定）。

### 19.5 最小改动清单（登记，不立即实现）

| # | 文件 | 改动 | 状态 |
| --- | --- | --- | --- |
| 1 | `assets/robots/unitree.py` | `UNITREE_MODEL_DIR` 指向本地资产 | ✅ **已改**（环境搭建时） |
| 2 | `go2/velocity_env_cfg.py` | 命令域已符合决策，**无需改**；如需补延迟随机化/身高终止再加两项 | 待用户定 |
| 3 | `scripts/rsl_rl/play.py` | 已有 `try/except` 兼容分支，**无需改** | 无需 |
| 4 | 新增评估工具 | 速度误差、响应时间、停车漂移、失败率 | TASK-020 |
| 5 | 新增外部命令接口 | 10 Hz 三维速度输入 | TASK-018 |
| 6 | 新增 CaMRA 训练配置副本 | **不改上游默认**，另建配置类 | TASK-019 |

### 19.6 接手最短路径

1. 读本笔记顺序：§1 版本基线 → §10 差异登记 → §13 频率链 → §16 观测/动作 → §18 导出。
2. 关键源码：`scripts/rsl_rl/train.py`、`go2/velocity_env_cfg.py`、`assets/robots/unitree.py`、
   `utils/export_deploy_cfg.py`、`tasks/locomotion/robots/go2/__init__.py`。
3. 运行环境：Tiger12，`/pub/data/caohy/miniconda/envs/env_isaaclab`，tmux 会话 `camra-locomotion`。
4. 下一步：从 TASK-013 起按依赖顺序推进。

### 19.7 导航成本与 footprint 的处置

本阶段**只保存 Go2 资产几何（USD）与版本**，不定义导航成本函数与 footprint。
参考：1020 稿的代理成本 `dt·(3.6+3·v²+20·(v−u)²+2·max(0,0.45−clearance))` 与
`simulation_footprint: 圆盘 r=0.20 m`、`hardware_footprint: 0.70 m × 0.32 m 矩形`，
均属**合成设定**，留待 navigation 阶段重新定义，不得当作实测成本。

## 20. 目标设备与资源边界（TASK-013）

### 20.1 设备事实

| 项 | 值 |
| --- | --- |
| 主机 | `tiger12.ios.ac.cn`（192.168.20.12），Ubuntu 24.04.5，内核 6.8.0-147 |
| CPU / 内存 | 96 逻辑核；503 GB RAM（可用 ≈283 GB） |
| GPU（6 张） | 索引 0/1/2/5 = **L40 46 GB**；索引 3/4 = RTX PRO 6000 Blackwell（**sm_120，torch 2.7 cu126 不支持**） |
| 可用 L40 空闲显存 | GPU0 ≈22.4 GB、GPU2 ≈22.4 GB、GPU5 ≈30.7 GB、GPU1 ≈6.5 GB |
| 驱动 / CUDA | 580.178.04 / CUDA 13.0（驱动侧） |
| Python 环境 | `/pub/data/caohy/miniconda/envs/env_isaaclab`，Python 3.11.18 |
| 软件栈 | isaacsim 5.1.0.0、isaaclab 0.47.2（Isaac Lab 2.3.0）、rsl-rl-lib 3.0.1、torch 2.7.0、numpy 1.26.0 |
| 磁盘 | `/` 63%（156G 空闲）、`/pub` 22T 剩 17T、`/home` 18% |
| 调度器 | **无**（无 sbatch/qsub/bsub）；cgroup 无 cpu.max / memory.max 限制 |
| 权限 | uid 39400170，`/pub/data/caohy` 可写 |
| tmux | 3.4；会话 `camra-locomotion`，窗口 `TASK-001` / `TASK-013` |
| 路径映射 | `remote_workspace=/home/caohy/repositories/Paper`、`run_root=/pub/data/caohy`、资产 `/pub/data/caohy/unitree_model` |

远程代码版本：根仓库 `8bed6d5`（main），HiRA 子模块 `6a05af5` 可用且干净。
断线保活：本任务期间多次 SSH 断开，tmux 窗口内的基准进程持续运行并正常结束（EXIT=0）。

### 20.2 关键阻塞发现：`debug_vis=True` 触发 300 秒网络超时

首次基准**卡死 7 分钟且 GPU 利用率 0%**。定位过程：加阶段标记发现已越过 `gym.make` 的场景创建
（地形 0.10 s、场景 0.61 s、仿真启动 0.61 s），随即停住；`py-spy` 采样与日志显示：

```
[isaaclab.utils.assets] Checking server availability for USD path:
  https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/Props/UIElements/arrow_x.usd
  (timeout: 300s)
[isaaclab.utils.assets] Waiting for server response... (30s elapsed)
```

根因：`CommandsCfg.base_velocity.debug_vis = True` 会去 S3 拉 `arrow_x.usd` 画命令箭头，
集群无外网 → **每次等待 300 s**。设为 `debug_vis=False` 后 `gym.make` 降到 **2.2–9.6 s**。

**处置**：headless 训练与评估一律关闭命令可视化；这是 TASK-019 训练配置的必改项，非可选优化。

### 20.3 吞吐与显存实测（L40，cuda:0，仅仿真步进，无训练更新）

| num_envs | make_s | policy 步/秒 | 每步 ms | transitions/秒 | 相对实时 | 本进程显存 |
| --- | --- | --- | --- | --- | --- | --- |
| 512 | 2.15 | 28.1 | 35.6 | 14,399 | 0.56× | — |
| 1024 | 3.20 | 19.1 | 52.3 | 19,566 | 0.38× | — |
| 2048 | 4.70 | 11.7 | 85.4 | 23,989 | 0.23× | — |
| **4096** | **8.74** | **6.8** | **146.1** | **27,970** | **0.14×** | **≈8.1 GB**（GPU0 上 5720+426+426+554+554+426 MiB） |

4096 档并发采样：**GPU0 利用率 87–90%**，说明物理与图形均在 GPU 上执行（非 CPU 回退）。
`torch.cuda.max_memory_reserved()` 仅 0.09 GB，**不能代表真实显存占用**（PhysX/Kit 不经过 torch 分配器），
必须以 `nvidia-smi` 的 per-PID 统计为准。

### 20.4 三项决策（基于上述数据）

**决策 A：并行环境数 —— 保留 4096，不下调。**
理由：4096 档显存仅 ≈8.1 GB（GPU0 46 GB、空闲 22 GB），显存不是约束；transitions/s 在四档中最高（27,970）。
调整依据是吞吐饱和而非显存。

**决策 B：训练预算 —— 建议 2000–3000 迭代/种子、3 个种子。**
推算：每迭代 `4096 × 24 = 98,304` transitions，采集耗时 `98,304 / 27,970 ≈ 3.5 s`。

| 迭代数 | 纯采集耗时 | 含更新开销（按 1.5–2× 估） |
| --- | --- | --- |
| 1000（1020 稿设定） | 0.98 h | 1.5–2 h |
| **2000（建议下限）** | **1.95 h** | **3–4 h** |
| **3000（建议上限）** | **2.93 h** | **4.5–6 h** |
| 50000（上游默认） | 48.8 h | 73–98 h ✗ 不可接受 |

3 个种子 × 2000–3000 迭代 ≈ **9–18 h**，可夜间完成。
上游默认 50000 迭代在本设备不可行，必须显式下调（TASK-019 写入配置）。

**决策 C：评估样本量 —— 不受算力限制，可采论文规模 900 trials 甚至更多。**
批量评估 4096 环境并行：一个 20 s 回合 = 1000 个策略步 × 0.146 s ≈ 146 s，即 **4096 回合/146 s**；
900 trials ≈ **32 s**，10,000 trials ≈ 6 分钟。因此样本量不是瓶颈，评估应取 3 种子 × 3 场景 × 100 trials = 900 起。

**性能阈值本身无法由吞吐数据决定**（那是质量目标而非资源），下表取值已由用户在 2026-10-31 采纳，TASK-017 直接据此固定，不再重新征询。

### 20.5 评估阈值（**已确认，用户 2026-10-31 采纳**）

| 指标 | 建议阈值 | 依据 |
| --- | --- | --- |
| 前向速度跟踪 RMS 误差 | ≤ **0.15 m/s** | 命令域上限 1.0 m/s 的 15% |
| 侧向速度 RMS 误差 | ≤ **0.10 m/s** | 命令域上限 0.4 m/s 的 25% |
| 偏航角速度 RMS 误差 | ≤ **0.15 rad/s** | 命令域上限 1.0 rad/s 的 15% |
| 阶跃响应时间（达 90%） | ≤ **0.5 s** | 10 个策略周期（50 Hz） |
| 停车漂移（停后 2 s 位移） | ≤ **0.10 m** | 约为机身半长 |
| 站立姿态漂移（10 s） | ≤ **0.05 m** | 静止抖动上限 |
| 失败率（跌倒/非法接触） | ≤ **5%**（常态）/ ≤ **10%**（含扰动与变速） | 20 s 回合、0.8 rad 倾角终止判据 |

### 20.6 三项决策的确认状态

| 决策 | 结论 | 状态 |
| --- | --- | --- |
| A 并行环境数 | 保留 **4096**（显存 ≈8.1 GB，非瓶颈；吞吐 27,970 transitions/s 最高） | **已确认**（用户 2026-10-31） |
| B 训练预算 | **2000–3000 迭代/种子 × 3 种子**（≈9–18 h）；上游默认 50000 迭代不可行，须下调 | **已确认**（用户 2026-10-31） |
| C 评估样本量 | **900 trials 起**（批量并行 900 trials ≈ 32 s），可按需扩大 | **已确认**（用户 2026-10-31） |
| 评估阈值 | 见 20.5 表（vx 0.15 m/s、vy 0.10 m/s、ωz 0.15 rad/s、响应 0.5 s、停车漂移 0.10 m、站立漂移 0.05 m、失败率 5%/10%） | **已确认**，TASK-017 直接采用 |

## 21. 兼容运行环境与 Go2 资产（TASK-014）

### 21.1 环境版本清单（实测）

| 组件 | 版本 | 备注 |
| --- | --- | --- |
| Python | 3.11.18 | conda 环境 `env_isaaclab` |
| isaacsim | **5.1.0.0** | 与计划基线一致 |
| isaacsim-extscache-kit / -kit-sdk / -physics | **5.1.0.0** | 三个均在（Tiger5 缺失即此三项，见 §10） |
| isaaclab / _assets / _tasks / _rl | 0.47.2 / 0.2.3 / 0.11.6 / 0.4.4 | Isaac Lab **2.3.0** 源码 editable |
| **rsl-rl-lib** | **3.0.1** | 2.3.0 官方钉版（非 5.0.1） |
| torch / torchvision | 2.7.0 / 0.22.0 | +cu126 |
| numpy | 1.26.0 | isaacsim 钉版 |
| tensordict / triton | 0.7.2 / 3.3.0 | 与 torch 2.7 兼容 |
| onnx / onnxscript | 1.23.2 / 0.7.2 | 导出 ONNX 用 |
| unitree_rl_lab | 0.2.1（editable） | 指向仓库 `CaMRA/locomotion/source/unitree_rl_lab` |

完整清单：`/pub/data/caohy/logs/camra/TASK-014/pip_freeze.txt`（248 个包）。

### 21.2 Go2 资产（本地化，含散列）

路径 `UNITREE_MODEL_DIR = /pub/data/caohy/unitree_model`，取自 `unitreerobotics/unitree_model`
的 LFS 媒体地址（上游 README 指向同名 HuggingFace 数据集，路径结构一致）。体积 18 MB。

| 文件 | SHA-256 |
| --- | --- |
| `Go2/usd/go2.usd` | `fd3c8c23c97cea3208fae1ab5eeacba4ffa616fe5f1397be6c7edf4feb93f180` |
| `Go2/usd/configuration/go2_description_base.usd` | `eeed42fb9e41395c47bc3fac7ff5d58625ac62f44bfa5a09cacebff83229c7a0` |
| `Go2/usd/configuration/go2_description_physics.usd` | `43800f436b3533c869f0b7774e5c70bc4291579df70a62263f808fb1a8f05505` |
| `Go2/usd/configuration/go2_description_sensor.usd` | `d63cfa5964eb5a72b8e2a02dc89448881497717f21d76b1ec9683a92cb765a16` |

仅配置了 Go2；**未修改其他机器人任务**（G1/H1/B2 的配置与路径保持上游状态）。

### 21.3 加载验证（真实运行，非命令罗列）

```
TASK014_CHECK {"app_ready_s": 1.39, "total_s": 8.25, "task": "Unitree-Go2-Velocity",
               "num_envs": 16, "obs_policy_dim": 45, "action_dim": 12,
               "step_dt": 0.02, "sim_dt": 0.005, "decimation": 4,
               "device": "NVIDIA L40", "robot_prim_exists": true}
```

- **观测 45 维、动作 12 维**：与 §16 的静态推导一致，此处为**运行时实测**。
- `sim_dt = 0.005`、`decimation = 4`、`step_dt = 0.02`：频率链 200/50 Hz 实测确认。
- 机器人 prim 存在，说明 USD 资产真正被加载（不只是路径可读）。
- `gym.make` 用时 1.39 s（关闭 `debug_vis` 后，见 §20.2）。

### 21.4 可复现安装步骤

脚本存放在 `/pub/data/caohy/scripts/`，按序执行即可重建：

| 步骤 | 脚本 | 内容 |
| --- | --- | --- |
| 1 | `01_miniconda.sh` | 安装 Miniconda 到 `/pub/data/caohy/miniconda`，建 `env_isaaclab`（py3.11），配 tuna 镜像 |
| 2 | `02_assets.sh` | 下载 4 个 Go2 USD 到 `/pub/data/caohy/unitree_model` |
| 3 | `03b_isaacsim.sh` | `pip install "isaacsim[all]==5.1.0.0"` + 补三个 extscache；缓存与 TMPDIR 指向 `/pub` |
| 4 | `04c_isaaclab.sh` | `pip install -e source/isaaclab` 等四个包 |
| 5 | `05_repin.sh` | 强制钉回 torch 2.7.0 / torchvision 0.22.0 / numpy 1.26.0 / click 8.1.7 / idna 3.10 / typing_extensions 4.12.2 / packaging 23.0 |
| 6 | `06_fix.sh` | `--no-deps` 装 tensordict 0.7.2、triton 3.3.0，补 orjson/onnx/GitPython |
| 7 | `10_unitree.sh` | 配置 `UNITREE_MODEL_DIR` 并 `pip install -e source/unitree_rl_lab` |

构建期易踩的三点（详见 §10、§20）：`setuptools==75.8.0` 才能构建 `flatdict`（当前已回到 84.2.1，
如需重建请先降级）；`rsl-rl-lib` 会顺带升级 torch，必须用 `--no-deps` 钉回；
`CUDA_VISIBLE_DEVICES` 会破坏 Kit 图形设备创建，改用 `CUDA_DEVICE_ORDER=PCI_BUS_ID` + `--device cuda:N`。

### 21.5 已知非阻塞问题

| 问题 | 影响 | 处置 |
| --- | --- | --- |
| `omni.iray.libs` 缺 `libGLU.so.1` | 仅离线电影级渲染（Iray），训练与 RTX 管线不受影响 | 记录，不处理 |
| `setuptools` 86→84.2.1 无 `pkg_resources` | 仅影响构建老式 setup.py 包 | 重建时降 75.8.0 |
| GPU 3/4 为 Blackwell sm_120 | torch 2.7.0+cu126 不支持 | 训练固定使用 L40（索引 0/1/2/5） |

## 22. 原始任务冒烟验证（TASK-015）

> 本记录是**基线冒烟**：证明任务可运行、契约正确、无基础设施错误。
> **不是控制器性能结果**——随机动作下机器人会摔倒，属未训练状态的预期现象。

### 22.1 运行条件

- `scripts/list_envs.py` 输出含 `Unitree-Go2-Velocity`（`LIST_ENVS_RC=0`）。
- 64 个并行环境、headless、L40（cuda:0）、随机动作 `0.6·N(0,1)`、连续 600 步（12 s 仿真）。
- `SMOKE_RC=0`，日志 `/pub/data/caohy/logs/camra/TASK-015/smoke.log`。

### 22.2 维度与配置（运行时实测）

| 项 | 实测值 |
| --- | --- |
| obs(policy) 每组维度 | **45** |
| obs(critic) 每组维度 | **60** |
| action_dim | **12** |
| action term | `JointPositionAction`（唯一动作项） |
| `sim.dt` / `decimation` / `step_dt` | 0.005 / 4 / **0.02** |
| `agent_cfg.clip_actions` | **None** |
| 动作项 `clip` | `(-100.0, 100.0)`（per-joint） |
| 动作 `scale` | **0.25**，`use_default_offset=True` |
| 关节位置限位（全部关节） | **[-2.723, 4.538] rad** |
| `stiffness` / `damping` | 25.0 ×12 / 0.5 ×12 |

### 22.3 受控输入核对动作链（关键证据）

给动作项喂固定 raw 值，读取 `processed_actions`（= 关节位置目标）：

| raw 动作 | 实测 target[rad] | 理论 `offset+0.25·raw` | clip 是否触发 |
| --- | --- | --- | --- |
| 0.0 | 0.100 | 0.100 | 否 |
| 0.5 | 0.225 | 0.225 | 否 |
| 10.0 | 2.600 | 2.600 | 否 |
| 1000.0 | **100.000** | 250.100 | **是** |

**结论（与 §16.2 一致，此处为实测）**：`clip ±100` 作用在**位置目标**上，需 `|raw| > 399.6` 才触发；
而关节限位只有 `[-2.72, 4.54] rad`。因此 **±100 完全不是有效动作范围**，真正的边界是关节限位 + 力矩包络 + PD。
若上层需要严格有界动作，必须显式设置 `agent_cfg.clip_actions`。

### 22.4 关节顺序（补充并修正 §16.4）

运行时打印两套顺序，**两者不同，且仿真侧还与直觉不同**：

```
sim joint_names  (按关节类型分组):
  FL_hip, FR_hip, RL_hip, RR_hip,
  FL_thigh, FR_thigh, RL_thigh, RR_thigh,
  FL_calf, FR_calf, RL_calf, RR_calf

joint_sdk_names  (按腿分组):
  FR_hip, FR_thigh, FR_calf, FL_hip, FL_thigh, FL_calf,
  RR_hip, RR_thigh, RR_calf, RL_hip, RL_thigh, RL_calf
```

即仿真内部是 **hip 组 → thigh 组 → calf 组**，SDK 是 **每条腿 hip/thigh/calf 连续**。
两者都不是简单反序，因此**导出时必须依赖 `joint_ids_map` 重排**（§18.2），任何手工按名字拼接都会错位。
§16.4 之前只记录了 SDK 顺序，此处补齐仿真侧顺序。

### 22.5 有限性、终止与重置

| 检查 | 结果 |
| --- | --- |
| 600 步内 obs 出现非有限值次数 | **0** |
| 600 步内 reward 出现非有限值次数 | **0** |
| 终止事件累计 | **16**（随机策略摔倒触发 `bad_orientation` / `base_contact`） |
| 重置后 obs 有限性 | True |
| 重置后 obs 形状 | (64, 45) |

终止与重置链路工作正常；摔倒属未训练预期，不是依赖或渲染问题。

### 22.6 吞吐与显存（64 环境）

42.1 policy 步/秒、20.4 ms/步（≈0.84× 实时）。`torch.cuda.max_memory_allocated()` 仅 0.009 GiB，
**再次说明该指标不适用于本栈**（见 §20.3），真实显存须用 `nvidia-smi`。

### 22.7 已知报错归类

| 报错 | 归类 | 影响 |
| --- | --- | --- |
| `omni.iray.libs` 缺 `libGLU.so.1` | 依赖缺失（Iray 离线渲染） | **与训练/本任务无关**，不阻塞 |
| `USD stage detach not called` / `Recursive unloadAllPlugins` | 退出期告警 | 不影响结果 |

### 22.8 未覆盖项（转交后续任务）

- **少量训练更新未在本次执行**：`train.py` 走同一份 `env_cfg`，其中 `debug_vis=True` 会触发
  300 s 网络超时（§20.2），必须先在 TASK-019 关闭；训练链路的端到端验证归 TASK-022。
- 4096 环境的吞吐已在 TASK-013 单独测得（6.8 policy Hz / 27,970 transitions/s），本任务不重复。

## 23. 底层与外部速度命令接口契约（TASK-016）

**可检查契约文件**：`CaMRA/locomotion/docs/command_interface.json`（v1.0.0，21 个顶层节，
含 27 个校准日志字段）。后续 TASK-018 实现、TASK-021 验证均以该文件为判据。

### 23.1 时间契约

| 层级 | 周期 | 频率 | 依据 |
| --- | --- | --- | --- |
| 物理仿真 | 0.005 s | **200 Hz** | `SimulationCfg.dt` |
| 底层推理 | 0.02 s（4 个物理步） | **50 Hz** | `decimation = 4` |
| 外部命令 | 0.1 s（5 次底层推理） | **10 Hz** | `high_level_action_repeat = 5`（§13.1） |

**保持语义（必须实现）**：一条外部命令在 0.1 s 内被**写入 5 次**，即在每次底层推理前
重写 `env.commands[:, :3]`。原因：底层命令项每 10 s 随机重采样，若不重写，命令会在 0.1 s 窗口内漂移
（HiRA 正是这么做的，见 §13.2）。**禁止**只在 5 次推理的开头写一次。

### 23.2 坐标、单位、批量

- 坐标系：**机身系**（body frame），`+x 前、+y 左、+z 上`。
- 单位：线速度 **m/s**，角速度 **rad/s**。
- 顺序固定为 `[vx, vy, wz]`；批量形状 `(num_envs, 3)`，`float32`，`cuda`。
- **`wz` 是偏航角速度，不是目标航向角**；`vy` **不得丢弃**（这两条列为禁止项）。

### 23.3 速度范围

| 集合 | vx | vy | wz |
| --- | --- | --- | --- |
| **limit（接口硬边界）** | ±1.0 | ±0.4 | ±1.0 |
| curriculum 初始 | ±0.1 | ±0.1 | ±1.0 |

外部接口以 **limit** 为硬边界；curriculum 只在训练中由初始值逐步放大到 limit。
取值来源为上游 `limit_ranges`（**用户 2026-10-31 已确认保留**，不采用 1020 稿的 0.6/0.4/1.2）。

### 23.4 观测契约（运行时已验）

45 维、六段固定顺序：`base_ang_vel(3, ×0.2)` → `projected_gravity(3)` → `velocity_commands(3)` →
`joint_pos_rel(12)` → `joint_vel_rel(12, ×0.05)` → `last_action(12)`。
critic 为 60 维（+`base_lin_vel` +`joint_effort`）。actor **不含任何仅仿真可得量**。
噪声仅训练期注入（`enable_corruption`），评估与导出必须关闭。

### 23.5 动作契约与真实边界

`q_des = q_default + 0.25 · a`，随后按动作项 `clip ∈ [-100, 100]` 裁剪。`agent_cfg.clip_actions = None`。

TASK-015 受控输入实测：`raw=1000 → 期望 250.1 → 实得 100.0`（裁剪触发），而
`raw=10 → 2.600`（未触发）。**裁剪需 `|raw| > 399.6` 才发生**，而关节限位仅 `[-2.723, 4.538] rad`。
结论：`±100` **不是有效动作范围**，真实边界是关节限位 + 力矩-速度包络 + PD(25/0.5)。
若上层需要严格有界动作，必须显式设 `clip_actions`，不得依赖 `±100`。

### 23.6 关节顺序与映射

| 来源 | 顺序 |
| --- | --- |
| 仿真 `joint_names` | 按**关节类型**分组：`FL_hip, FR_hip, RL_hip, RR_hip, FL_thigh, …` |
| SDK `joint_sdk_names` | 按**腿**分组：`FR_hip, FR_thigh, FR_calf, FL_hip, …` |

两者均非简单反序，**部署/导出必须使用 `joint_ids_map`**（`export_deploy_cfg`），禁止按名字手工拼接。

### 23.7 复位与状态清理

- 动作项 `reset()` 将 `_raw_actions` 置零 → `last_action` 观测在复位后为全零。
- **无观测历史**（`history_length = 0`），故无需清理历史缓冲。
- 命令在 reset 时重新采样。
- 回合 20 s；终止条件：`time_out`、`base_contact(>1.0)`、`bad_orientation(>0.8 rad)`。

### 23.8 命令来源切换与覆盖保护

外部命令接管后**必须压过随机课程**：每次底层推理前重写命令缓冲；并在日志中记录 `command_source`
（`curriculum` / `external`），便于事后区分。

### 23.9 无效命令处理与滤波政策

| 情形 | 处理 |
| --- | --- |
| `NaN` / `Inf` | 拒绝，保留上一条命令，`rejected` 计数 +1 |
| 超出 limit | 裁剪到 limit，`clamped` 计数 +1 |
| 形状不符 | 显式报错拒绝 |
| 丢弃 `vy` 或把 `wz` 当航向 | **禁止** |

**滤波/斜率限制**：本阶段**不启用**。若日后引入，必须同时记录 **原始命令** 与 **施加命令**，
且名义配对契约（calibration）以**施加命令**为准。

### 23.10 校准所需日志字段（27 项）

`sim_time_s`、`episode_id`、`env_index`、`cmd_raw_{vx,vy,wz}`、`cmd_applied_{vx,vy,wz}`、
`base_lin_vel_body_{x,y,z}`、`base_ang_vel_body_{x,y,z}`、`base_pos_world_{x,y,z}`、
`base_quat_world_{w,x,y,z}`、`joint_pos_12`、`joint_vel_12`、`terminated`、`truncated`、`command_source`。

采样频率 = 底层推理频率（50 Hz）。校准按**共同 0.1 s 网格线性插值**配对名义/执行轨迹（§18.4）。

### 23.11 与旧协议的差异取舍

| 项 | 1020 旧协议 | 本契约 | 取舍 |
| --- | --- | --- | --- |
| 高层周期 | 0.2 s（5 Hz），命令保持 10 次 tracker 更新 | **0.1 s（10 Hz），保持 5 次推理** | 采用新值；旧论文不改 |
| 观测维度 | 48（含机身线速度） | **45**（不含） | 保留 45，**不为凑 48 新增传感量** |
| PD 刚度 | 20.0 | **25.0** | 保留 25，**不混用旧增益** |
| 命令域 | 0.6 / 0.4 / 1.2 | **1.0 / 0.4 / 1.0** | 采用代码值（用户已定） |
| 动作裁剪 | `q_stand + 0.25·clip(a,−1,1)` | 目标级 `±100` + `clip_actions=None` | 记录现状；**不擅自改成 ±1** |

**明确声明**：本契约未为对齐示例协议盲目增加输入维度，也未修改 PD 增益；45 维与 25/0.5 均为实测确认的现状基线。

## 24. 评估协议与验收条件（TASK-017，训练前固定）

**可检查协议文件**：`CaMRA/locomotion/docs/eval_protocol.json`（v1.0.0）。
本协议在**正式训练之前**固定，训练后不得因结果不理想而修改阈值。

### 24.1 种子划分（互斥）

| 用途 | 种子 | 说明 |
| --- | --- | --- |
| 训练 | `0, 1, 2` | 3 个种子（TASK-013 决策 B） |
| 开发/选模 | `100–109` | 用于 checkpoint 选择，**不参与**最终报告 |
| 最终测试 | `200–209` | 选定的 checkpoint **只评一次** |

地图与命令程序由 `(seed, scenario, episode_index)` 派生，保证可复现。

### 24.2 场景与阈值（10 个）

| 场景 | 命令程序 | 主要指标 → 阈值 |
| --- | --- | --- |
| `stand` | 0,0,0 / 10 s | 平面漂移 ≤ **0.05 m**；roll、pitch ≤ **8°** |
| `fwd_low` | vx=0.3 | rmse_vx ≤ **0.15**；p95 ≤ 0.25 |
| `fwd_mid` | vx=0.6 | rmse_vx ≤ **0.15**；p95 ≤ 0.25 |
| `fwd_high` | vx=1.0（上限） | rmse_vx ≤ **0.18**；p95 ≤ 0.30 |
| `backward` | vx=−0.5 | rmse_vx ≤ **0.15**；p95 ≤ 0.25 |
| `lateral` | vy=±0.3 | rmse_vy ≤ **0.10**；p95 ≤ 0.18 |
| `turn_inplace` | wz=±0.8 | rmse_wz ≤ **0.15**；p95 ≤ 0.25 |
| `combined` | (0.6, ±0.3, ±0.5)，4 组合 | 三项 RMS 分别沿用单项阈值 |
| `start_stop` | 0→0.8→0，12 s | 上升时间 ≤ **0.5 s**；停车漂移 ≤ **0.10 m**；超调 ≤ 0.15 m/s |
| `ramp` | vx 0→1.0→0，14 s | rmse_vx ≤ **0.18**；p95 ≤ 0.30 |

阈值来源：前七项与 `stand`、`start_stop` 的主阈值来自 TASK-013 实测推导并**经用户 2026-10-31 采纳**；
`fwd_high` / `ramp` 的 0.18 m/s 与 `start_stop` 超调 0.15 m/s 为本次新增，理由已写入协议 `source` 字段
（上限场景放宽 0.03 m/s；超调取命令幅值 0.8 的 19%）。**没有任何阈值来自论文示例数字。**

### 24.3 指标定义（全部可计算，不依赖主观判断）

`rmse_*`、`p95_abs_err_*`、`rise_time_s`（达 90% 且维持 0.2 s）、`stop_drift_m`（命令归零后 2 s 平面位移）、
`overshoot_mps`、`stand_drift_m`、`max_roll_deg` / `max_pitch_deg`、`failure_rate`、
`action_rate_rms`、`joint_limit_rate`。逐项定义见协议 `metric_definitions`。
所有跟踪指标在 reset 后 **1.0 s 稳态窗口**之后统计。

### 24.4 标称与扰动分组（只用已实现的旋钮）

| 分组 | 摩擦 | 附加质量 | 推力 | 观测噪声 |
| --- | --- | --- | --- | --- |
| `nominal` | 固定 1.0 | 0 | 关闭 | 关闭 |
| `perturbed` | (0.3, 1.25) | (−1, 3) kg | ±0.5 m/s，每 5–10 s | 开启（训练值） |

⚠️ **命令延迟随机化（0–40 ms）当前环境未实现，协议明确不覆盖、不得声称已验证**（见 §17.3 缺失项）。
扰动组的 rmse 阈值放宽 20%，失败率上限放宽到 0.10。

### 24.5 回合预算

每个场景 × 每个训练种子 **100 回合** → 每场景 300 → 每组 3000 → **两组共 6000 回合**。
按 TASK-013 实测的批量并行能力，6000 个 10–14 s 回合约 **4 分钟**（单张 L40）。
评估算力充裕，因此不压缩样本量（远高于论文规模 900）。

### 24.6 检查点选择与训练停止规则

| 项 | 规则 |
| --- | --- |
| 选模数据 | dev 种子 300 回合，每 100 次迭代评一次（500→3000） |
| 选模分数 | `0.4·rmse_vx/T + 0.2·rmse_vy/T + 0.2·rmse_wz/T + 0.2·failure/T`（T 为各自阈值） |
| 约束 | dev 失败率 ≤ 0.10，在满足者中取分数最小 |
| 兜底 | 若无一满足，取失败率最小者并**记为验收失败** |
| 最大迭代 | **3000**（TASK-013 决策 B） |
| 墙钟上限 | **6 h/种子**，超时即停止并保留最佳 dev checkpoint |
| 早停 | dev 分数连续 500 次迭代无改善 |
| 重训规则 | 每种子**至多 1 次**，且仅限基础设施错误（NaN/崩溃/资源丢失）；**性能不达标不得重训** |

### 24.7 通过判定

- **逐场景**：该场景 300 回合聚合后全部阈值成立。
- **整体**：nominal 组每个场景都通过；perturbed 组通过率 ≥ 80%；两组全局失败率分别 ≤ 0.05 / 0.10。
- **不可作为证据**：视频播放、回合总奖励、课程进度、训练损失。**只看这些一律不通过。**

### 24.8 报告产物

逐场景表（含每种子数值以暴露种子方差）、逐回合 CSV、汇总 JSON、checkpoint 散列、`env.yaml`/`agent.yaml`、
资产 sha256，统一写到 `/pub/data/caohy/logs/camra/TASK-024/`。

## 25. 外部速度指令接口（TASK-018）

**实现文件**（新增，未改动上游任何文件）：

| 文件 | 行数 | 职责 |
| --- | --- | --- |
| `source/unitree_rl_lab/unitree_rl_lab/utils/external_command.py` | 745 | 命令程序、命令桥接、gym wrapper |
| `scripts/external_command_drive.py` | 264 | 独立驱动入口（输出实际施加命令 + 仿真时间的 CSV） |

契约来源为 §22 的 `command_interface.json` v1.0.0。

### 25.1 注入点：为什么必须在 `term.compute()` 之后

`ManagerBasedRLEnv.step()` 的内部顺序（`manager_based_rl_env.py` L182–235）是：

```
物理步进(decimation 次) → episode_length_buf += 1 → 终止判定
→ _reset_idx(自动复位) → command_manager.compute(step_dt) → observation_manager.compute()
```

命令计算在**复位之后、观测之前**。这决定了三件事：

1. 在 `step()` **之前**写命令不够：`compute()` 可能把它重采样掉。
2. 在 `step()` **之后**写命令也不够：本步的观测已经用旧命令算完了。
3. 唯一正确的位置是**包住 `term.compute()`**：先执行原生计算，立刻写回外部值，随后观测才被组装。

实现即 `ExternalCommandBridge._install()` 中的 `locked_compute`。这也是"每次推理前重写命令缓冲"的具体落点。

### 25.2 三类覆盖风险与对应禁用

| 风险 | 机制 | 处置 |
| --- | --- | --- |
| 定时重采样 | `CommandTerm.compute` 的 `time_left <= 0` → `_resample()` | `cfg.resampling_time_range=(1e9,1e9)` + `time_left.fill_(1e9)` |
| 站立环境清零 | `_update_command()` 对 `is_standing_env` 写 0 | `cfg.rel_standing_envs=0` + `is_standing_env.zero_()` |
| 航向控制改写 wz | `heading_command=True` 时用 heading 误差覆盖 wz | `cfg.heading_command=False` |

`close()` 会逐项恢复原值并重新 arm `time_left`，因此**训练路径不受影响**（不构造桥接就完全不变）。

### 25.3 复位语义：新回合不继承旧命令

IsaacLab 在 `step()` 内部自动复位终止的环境，返回的观测已经是新轨迹第一帧。若沿用旧命令，等于把两条轨迹缝合。

判定方法：`episode_length_buf` 在 `compute()` 之前已 `+1`，而被 `_reset_idx` 复位的环境归零，因此
**`buf == 0 且上一步 buf != 0` ⇔ 本步发生了自动复位**。桥接在 `locked_compute` 中据此清空这些环境的命令
（`on_reset="zero"`，可改 `"hold"`），清空发生在观测组装之前。

同时 `CommandStepRecord` 逐环境返回 `terminated` / `truncated` / `new_episode` / `episode_index` / `step_index` / `command_time`，
并提供 `trajectory_valid_mask()`（= `~new_episode`），供调用方剔除跨回合样本。

### 25.4 命令程序与时间基准

| 程序 | 参数 | 说明 |
| --- | --- | --- |
| `fixed` | `vel` | 恒定命令 |
| `step` | `segments=[{duration, vel}]` | 分段常值，末段保持 |
| `ramp` | `start,end,duration,return_to_start,hold` | 线性变化，可选回落 |
| `sine` | `axis,amplitude,frequency,base,phase` | 单通道连续变化 |
| `combo` | `segments=[{duration,type,kwargs}]` | 上述程序的任意组合 |

**时间基准默认 `time_source="episode"`**，即每个环境**当前回合内**的时间（`episode_length_buf × step_dt`）。
理由：评估与校准以回合为单位，长跑后若用全局时钟，程序早已走完（本任务首轮验证即踩到该问题）。
需要跨回合连续变化时可切 `time_source="global"`。

`CommandStepRecord.sim_time` 始终是**全局仿真时间**，与 `command_time` 分开记录，符合契约"输出实际施加命令和仿真时间"。

### 25.5 契约验收证据（`/pub/data/caohy/logs/camra/TASK-018/verify.json`）

主机 tiger12 / 会话 `camra-locomotion` / 窗口 `TASK-018`，`num_envs=4`、`step_dt=0.02`、`all_pass=true`。

| 检查 | 结果 | 关键数值 |
| --- | --- | --- |
| A 安装即禁用内部生成器 | PASS | `resampling=(1e9,1e9)`、`rel_standing=0.0`、`heading_command=False`、`is_standing_env` 全 False |
| B 跨 10 s 重采样边界仍保持 | PASS | 15.00 s / 750 步，`max_abs_dev=0.0`，**定时重采样次数 0** |
| C 程序按步网格更新 | PASS | 4 段命令，切换步 `[51,101,151]`，间隔 `[51,50,50]`（期望 50±2） |
| D 超限请求被裁到契约上限 | PASS | 请求 `[2.0,-2.0,5.0]` → 施加 `[1.0,-0.4,1.0]`，`clipped=True` |
| E 非有限请求被拒 | PASS | `[0.5,nan,0.0]` → 施加 `[0,0,0]`，`invalid=True` |
| F 复位被报告且命令清零 | PASS | `new_episode=True`、`truncated=True`、复位后命令 0.0、`episode_index 2→3`、`command_time` 归零 |
| I 观测反映实际施加命令 | PASS | `applied=[0.35,-0.15,0.45]` 与 `obs["policy"][0,6:9]` **逐位相同** |
| G 关闭桥接恢复训练配置 | PASS | `resampling` 回到 `(10.0,10.0)`、`rel_standing_envs` 回到 0.1 |
| H 训练模式重新按定时重采样 | PASS | 关闭后第 499 步（≈9.98 s）命令自行变化 |

**驱动入口实测**（`logs/camra/TASK-018/drive_{step,combo}.csv`）：

- `step` 程序 12 s / 600 步 / 4 环境 → 2400 行 CSV，命令在 3.04 / 6.04 / 9.04 s 精确切换（每段 150 步 = 3 s）。
- `combo` 程序 8 s → 400 步，`fixed → ramp → sine(wz) → fixed` 链式调用正常，末帧 `wz=-0.0157` 与正弦相位一致。
- 两次运行 `terminated=0`、`truncated=0`（12 s 内未触发 20 s 上限），`bridge_stats.internal_resample_count` 的增量全部来自 `_reset_idx`，且都被 `locked_compute` 立即覆盖。

### 25.6 用法

```python
from unitree_rl_lab.utils.external_command import ExternalCommandWrapper, FixedProgram

env = gym.make("Unitree-Go2-Velocity", cfg=env_cfg)
env = ExternalCommandWrapper(env, program=FixedProgram([0.8, 0.0, 0.0]), on_reset="zero")
obs, _ = env.reset()
obs, rew, term, trunc, info = env.step(actions)
rec = info["external_command"]
valid = rec.trajectory_valid_mask()   # 剔除本步自动复位的环境
```

命令行：`python scripts/external_command_drive.py --headless --num_envs 4 --program step --episode_seconds 12`

### 25.7 已知边界

- **未实现命令延迟随机化**（0–40 ms），与 §24.4 一致，仍属不覆盖项。
- 自动复位后的新回合命令默认清零；若调用方希望续用，应显式 `set_command(_)`，不要依赖 `on_reset="hold"` 的隐式行为。
- 桥接只接管 `base_velocity` 一个命令项；将来若增加命令项需分别桥接。
- 本任务不涉及 navigation，也未复制 HiRA 框架（仅借鉴其"每步重写命令"的经验）。

## 26. CaMRA 底层训练配置（TASK-019）

采用**覆盖层**而非改写上游：`velocity_env_cfg.py` 一字未动，CaMRA 配置继承它并只改下表列出的项。
这样上游基线仍可供任何其他机器人任务使用，新旧差异就是这张表本身。

| 文件 | 作用 |
| --- | --- |
| `tasks/locomotion/robots/go2/camra_env_cfg.py` | 环境覆盖层 + 可审查差异表 + 运行时断言 |
| `tasks/locomotion/agents/camra_ppo_cfg.py` | PPO runner 覆盖层 + 差异表 |
| `tasks/locomotion/robots/go2/__init__.py` | 追加注册 `CaMRA-Go2-Velocity`（原任务保留） |

### 26.1 环境侧：**只改两项**

| 项 | 上游 | CaMRA | 目的 |
| --- | --- | --- | --- |
| `commands.base_velocity.debug_vis` | `True` | **`False`** | 关掉命令箭头可视化。开着时 `isaaclab.utils.assets` 要去 S3 解析 `arrow_x.usd`，集群无外网 → **每次建环境等 300 s**（TASK-013 实测：`gym.make` >300 s 且 GPU 利用率 0%）。关掉后实测 4.98 s。 |
| `commands.base_velocity.resampling_time_range` | `(10.0, 10.0)` | **`(4.0, 8.0)`** | 20 s 回合内命令切换次数从 2 次增到 3–5 次，覆盖评估协议里 `start_stop` / `combined` 需要的**命令过渡**。保持时间 4–8 s 比 10 Hz 外部接口慢两个数量级，仍是**分段常值**，不是每步白噪声。 |

改动影响面：两项都不触碰观测、动作、奖励、终止与扰动定义；**命令域仍是 ±1.0 / ±0.4 / ±1.0**。
回退方式：`CAMRA_COMMAND_RESAMPLING = (10.0, 10.0)`、`debug_vis=True`。

### 26.2 明确**不改**的项（写下来以便审计）

| 项 | 现状 | 为什么不动 |
| --- | --- | --- |
| `scene.num_envs` | 4096 | TASK-013 实测 4096 环境仅占 ~8.1 GB 显存（L40 46 GB），四档里吞吐最高（27,970 transitions/s），无需下调 |
| `scene.terrain` | 只有 `flat` | 上游其余 `sub_terrains` 全部注释掉，任务本身已限定平地，无需额外限制 |
| `limit_ranges` | ±1.0 / ±0.4 / ±1.0 | 用户 2026-10-31 决策保留代码值 |
| 奖励 | 16 项权重全保持 | `track_lin_vel_xy` 1.5、`track_ang_vel_z` 0.75 等不动，避免"换奖励掩盖问题" |
| 域随机化 | 摩擦 0.3–1.2、附加质量 −1~3 kg、推力 0.5 m/s 每 5–10 s | 与 1020 稿相比已更强，保持 |
| 终止集合 | 保持 | **身高终止（<0.18 m）与命令延迟随机化（0–40 ms）仍未加入**：两者都会改变 §24 已固定的失败率口径，属明确非目标（见 §17.3） |
| `sim.dt` / `decimation` | 0.005 / 4 | 200 Hz 物理、50 Hz 策略，与契约一致 |

### 26.3 智能体侧：**只改三项**

| 项 | 上游 | CaMRA | 目的 |
| --- | --- | --- | --- |
| `max_iterations` | 50000 | **3000** | 实测每次迭代采集约 3.5 s，50000 次纯采集需 ~48.8 h/种子，本机跑不完；3000 是 §24.6 固定的上限（含 6 h 墙钟与早停 500） |
| `experiment_name` | `""`（回落到任务名） | **`camra_go2_locomotion`** | 固定日志根，三个种子与评估产物归一个可审计目录 |
| `seed` | 未设 | **0** | §24.1 固定训练种子 {0,1,2}；默认 0 让首跑可复现，另两个用 `--seed 1/2` 覆盖 |

网络结构（[512,256,128] ELU）、PPO 超参（lr 1e-3 adaptive、γ 0.99、λ 0.95、clip 0.2、5 epochs / 4 minibatches）、
`empirical_normalization=False` **全部继承未改**。

### 26.4 训练入口与日志目录

`train.py` 的日志根是相对当前目录的 `logs/rsl_rl/<experiment_name>`，因此**必须在固定 run 目录启动**：

```bash
mkdir -p /pub/data/caohy/run/camra_go2 && cd /pub/data/caohy/run/camra_go2
ln -sfn /pub/data/caohy/logs/rsl_rl logs          # 日志落在 /pub，不占 50 GB 的 home 配额
export OMNI_KIT_ACCEPT_EULA=YES XDG_DATA_HOME=/pub/data/caohy/xdgdata        XDG_CACHE_HOME=/pub/data/caohy/xdgcache TMPDIR=/pub/data/caohy/tmp        CUDA_DEVICE_ORDER=PCI_BUS_ID
unset CUDA_VISIBLE_DEVICES
python /home/caohy/repositories/Paper/CaMRA/locomotion/scripts/rsl_rl/train.py        --task CaMRA-Go2-Velocity --headless --seed 0
```

恢复训练：追加 `--resume --load_run <run_dir> --checkpoint <model_*.pt>`；`--max_iterations` 可在命令行临时覆盖。

### 26.5 验证证据（`/pub/data/caohy/logs/camra/TASK-019/verify.json`，`all_pass=true`）

主机 tiger12 / 会话 `camra-locomotion` / 窗口 `TASK-019`：

| 检查 | 结果 | 关键数值 |
| --- | --- | --- |
| A 任务注册 | PASS | 三个 entry point 齐全且指向 `camra_env_cfg` |
| B 环境配置差异**恰好两项** | PASS | `diff_keys = {debug_vis, resampling_time_range}`（对全量 `to_dict()` 做叶子级 diff） |
| C 运行时断言 | PASS | `assert_overrides_applied()` 两项均匹配（`False` / `(4.0, 8.0)`） |
| D 智能体配置差异**恰好三项** | PASS | `{experiment_name, max_iterations, seed}`，`max_iter=3000` |
| E 平地 + 并行量 | PASS | `sub_terrains={'flat': 0.1}`、默认 `num_envs=4096` |
| F 建环境不再被 S3 卡住 | PASS | `gym.make=4.98 s`（阈值 120 s），`debug_vis=False`，随后 20 步正常 |

B 与 D 是对**实际解析出的配置对象**做结构 diff，不是人工比对文档，因此"差异可审查"是可执行的。
`assert_overrides_applied()` 会在训练启动前抛出 `AssertionError`，防止代码与文档悄悄漂移。

### 26.6 与 TASK-018 接口的衔接

训练用内部课程（`base_velocity` 定时重采样）；评估/驱动用 `ExternalCommandWrapper` 接管同一命令项。
两者不会互相覆盖，已由 TASK-018 的 G/H 检查证明——桥接安装时禁用内部重采样，关闭后恢复。
评估时必须使用 `CaMRA-Go2-Velocity`（`debug_vis=False`），否则每次建环境都会等 300 s。

## 27. 底层评估与日志工具（TASK-020）

| 文件 | 行数 | 职责 |
| --- | --- | --- |
| `source/unitree_rl_lab/unitree_rl_lab/utils/eval_metrics.py` | 170 | 纯 NumPy 指标定义（无 Isaac 依赖），评估器与复算脚本共用 |
| `scripts/eval_locomotion.py` | 604 | 按固定协议跑场景、记录步级日志、输出逐回合与分组摘要 |
| `scripts/recompute_metrics.py` | 135 | 独立进程从原始 npz 复算指标并与 CSV 比对 |

指标公式只存在**一处**（`eval_metrics.py`），评估器与复算器都 import 它。这样"可由原始日志复算"才是可检验的，而不是两套实现互相印证。

### 27.1 确定性推理与观测约定

- 动作来自 `ActorCritic.act_inference`（**均值**动作，无采样噪声），不是 `act`。
- 传给它的是**完整 obs 字典**（`act_inference` 内部按 `obs_groups` 选组），不是 `obs["policy"]`。
- 推理包在 `torch.no_grad()` 里。

### 27.2 每步记录字段（56 项 / 50 Hz）

| 组 | 字段 | 说明 |
| --- | --- | --- |
| 时间 | `sim_time`, `episode_step` | 全局仿真秒、当前回合内步号 |
| 命令 | `cmd_vx/vy/wz`, `app_vx/vy/wz` | 请求值 vs 命令缓冲里的实际值（可检出裁剪） |
| 速度 | `vx_body`, `vy_body`, `wz_body` | **机身系**，`root_lin_vel_b` / `root_ang_vel_b[:,2]` |
| 姿态 | `roll`, `pitch`, `yaw` | 由 `root_quat_w` 换算 |
| 位姿 | `pos_x/y/z` | 世界系根位置 |
| 关节 | `joint_pos_0..11`, `joint_vel_0..11` | 仿真顺序（按关节类型分组，见 §16.4） |
| 动作 | `action_0..11` | 12 维原始动作（未乘 scale） |
| 终止 | `terminated`, `truncated`, `reason_id` | reason: 0 none / 1 base_contact / 2 bad_orientation / 3 time_out / 4 scenario_end |

采样频率 = 策略步 = 50 Hz；`step_dt` 写入 npz 与 `run_info.json`，复算时用同一值。

### 27.3 指标与统计口径

| 项 | 口径 |
| --- | --- |
| 误差坐标系 | 机身系（+x 前 / +y 左 / +z 上），m/s 与 rad/s |
| 稳态窗口 | reset 后**前 1.0 s 排除**（协议 `warmup.settle_s`）；若回合短于 1 s 则退化为全窗口并在数值上如实反映 |
| rmse / p95 | `sqrt(mean((actual-commanded)^2))`；`percentile(|actual-commanded|, 95)` |
| 起停 | 阶跃起点→首次达 90% 且维持 0.2 s 为上升时间；`max(vx)-幅值` 为超调；命令归零后 2.0 s 平面位移为漂移 |
| 失败 | `base_contact` 或 `bad_orientation` 终止 → `failed=1` |
| 动作抖动 | `mean_t ||a_t - a_{t-1}||_2`（12 维原始动作） |

**提前终止处理**：终止回合**不丢弃**，其终止前的样本照常参与所有指标；`failure_rate` 单独统计。协议要求"不选择性丢弃跌倒轨迹"，实现上没有任何过滤分支。

### 27.4 扰动分组（只用已实现的旋钮）

| 分组 | 摩擦 | 附加质量 | 推力 | 观测噪声 |
| --- | --- | --- | --- | --- |
| `nominal` | 固定 1.0 | 0 | `(0,0)` | 关闭 |
| `perturbed` | `(0.3, 1.25)` | `(-1, 3)` kg | `±0.5` m/s，每 5–10 s | 开启（训练值） |

实际生效值逐次写进 `run_info.json` 的 `group_applied`，不靠文档声明。
**命令延迟随机化（0–40 ms）仍未实现**，两组都不覆盖，报告中不得声称已验证。

### 27.5 输出结构

```
<out>/
  run_info.json        # checkpoint+sha256、协议版本+sha256、git commit、seed、分组实际值、
                       # step_dt/policy_hz/warmup、data_origin、免责声明
  protocol_used.json   # 本次使用的协议副本
  env_cfg.json / agent_cfg.json
  per_episode.csv      # 一回合一行，含全部指标
  summary.json         # 逐场景聚合 + 阈值判定 + 总体 pass
  raw/<group>/<sc>.npz # steps(float64, 已拼接) + lengths + fields + step_dt
  snippets/<group>/<sc>_epNNN.{csv,png}   # 代表回合，仅辅助检查
  recompute_check.json # 复算比对报告
```

`run_info.json` 明写 `data_origin: isaac_sim_simulation` 与"SIMULATION DATA ONLY — not real-robot measurements"；
本阶段**不计算 α/δ**（写在 `disclaimer` 里）。

### 27.6 验证证据

全部在 tiger12 / 会话 `camra-locomotion` / 窗口 `TASK-020` 内执行，日志在 `/pub/data/caohy/logs/camra/TASK-020/`。

**(1) 指标可复算** —— `recompute_metrics.py` 独立进程读 npz 重算，与 `per_episode.csv` 逐项比对：

```
[RECOMPUTE] fwd_mid     episodes=4 max_rel_err=0.000e+00
[RECOMPUTE] stand       episodes=4 max_rel_err=0.000e+00
[RECOMPUTE] start_stop  episodes=4 max_rel_err=0.000e+00
[RECOMPUTE] comparisons=204 mismatches=0 verdict=PASS
```

**(2) 同配置可复跑** —— 两次独立运行（`selftest` / `selftest_repeat`）逐项比对：

```
comparisons=204 beyond_1e-6=0 worst_rel=0.000e+00
VERDICT: PASS
summary identical: True
```

两次结果**逐位相同**，说明 seed 固定后评估链路完全确定。

**(3) 跌倒轨迹不被丢弃** —— `--action_mode random`（±4 raw，仅自检用）强制摔倒：

| 场景 | 回合 | 步数 | 结束原因 | failed |
| --- | --- | --- | --- | --- |
| stand | 4 | 17 / 41 / 51 / 54 | bad_orientation | 全部 1 |
| fwd_mid | 4 | 12 / 27 / 40 / 85 | bad_orientation | 全部 1 |

`failure_rate=1.000`、8 个回合一條不少地进了 CSV 与 npz，含 0.24 s 的超短回合。

**(4) 两个分组都可运行** —— `nominal` 与 `perturbed` 各自产出完整产物；
`perturbed` 的 `group_applied` 实测为摩擦 `[0.3,1.25]`、质量 `[-1,3]`、推力 `±0.5`、噪声开启。

**(5) 回放片段** —— 每场景 1 条，CSV（56 列原始步记录）+ PNG（vx/vy/wz 实际 vs 命令三联图），84–99 KB。

### 27.7 用法

```bash
python scripts/eval_locomotion.py --headless --task CaMRA-Go2-Velocity \
    --checkpoint <run>/model_3000.pt --num_envs 256 \
    --group nominal --scenarios all --episodes_per_scenario 300 \
    --seed 200 --out_dir /pub/data/caohy/logs/camra/TASK-024/nominal
python scripts/recompute_metrics.py --run_dir /pub/data/caohy/logs/camra/TASK-024/nominal
```

`agent_cfg` 优先从 checkpoint 同级的 `params/agent.yaml` 读取（保证网络结构与权重匹配），
缺失时回落到任务 entry point 并打印告警。

### 27.8 已知边界

- 本次自检使用的是**未训练策略**（`policy_source: untrained_init`），指标数值本身无意义，
  仅用于验证链路；正式数值待 TASK-023 产出 checkpoint 后由 TASK-024 生成。
- 失败回合在 warmup 内就结束时，指标含 warmup 期样本（无法再切窗）；这是如实反映，不做特殊处理。
- 视频未启用（headless 下渲染开销大）；回放以 CSV+PNG 曲线形式提供，符合"数值指标为主证据"。
- `episode_index` 是**每环境**的回合计数，不是全局序号；唯一键应为 `(scenario, episode_index, env_id)`。

## 28. 接口 / 节拍 / 指标验证（TASK-021）

| 文件 | 类型 | 依赖 |
| --- | --- | --- |
| `tests/test_units.py` | 21 项 pytest 单元测试 | 纯 CPU，不启仿真 |
| `scripts/verify_runtime_contracts.py` | 9 项运行时契约 | 需要 Isaac Sim |

所有夹具都是**合成的**，文件头写明 `SYNTHETIC_FIXTURE_NOTE`：这些测试只验证代码行为，
产生的任何数字都不得当作控制器性能结果。

### 28.1 单元测试（21/21 通过，`tests/test_units.py`）

| 类别 | 用例 |
| --- | --- |
| 轴与单位 | 命令通道顺序 `[vx,vy,wz]` 且数值原样透传；`SineProgram` 只在指定轴上叠加；`RampProgram` 端点为 0→1、斜率 0.5/s |
| 分段语义 | `StepProgram` 在段边界切换、末段保持 |
| 限幅 | 逐轴独立裁剪到 `[±1.0, ±0.4, ±1.0]`；恰在边界上的值不被改；非有限值置零并单独标记 |
| 无效输入 | 命令宽度≠3、空段列表、非法轴号、`duration=0` 均抛 `ValueError` |
| 指标算术 | 常值误差 0.1 → `rmse=p95=0.1`（解析值）；两级误差 ±0.1/−0.3 → `rmse=sqrt((0.01+0.09)/2)`；孤立 5.0 尖峰不影响 p95 但会抬高 rmse（说明 rmse 不鲁棒） |
| 窗口 | 前 1.0 s 的大误差不进入指标 |
| 终止统计 | 第 41 步终止 → `failed=1`、`ended_by=bad_orientation`、`steps_before_end=41`；`scenario_end` 不算失败 |
| 聚合 | 单回合聚合 = 该回合原值；批量取均值；均值**恰等于**阈值时判通过、超一点判失败 |
| 分组阈值 | perturbed 只把 rmse 放宽 1.2×（0.15→0.18），p95 与失败率不放宽；失败率上限 0.05 / 0.10 |
| 日志布局 | `NF==56`、字段名与切片的稳定契约（防 npz/CSV 静默错位） |

### 28.2 运行时契约（9/9 通过，tiger12 实测）

| 检查 | 结果 |
| --- | --- |
| A 速率链 200/50 Hz | 每个策略步恰好 **4** 个物理步；`step_dt=0.02 == 0.005×4` |
| A 10 Hz 间隔 | 每 0.1 s 恰好 **5** 个策略步（实测窗口命中） |
| B 时间戳 | 相邻 `sim_time` 增量全为 `step_dt`（浮点残差 <1e-6），单调无回退 |
| B 观测一致 | `obs["policy"][0,6:9]` 与施加命令逐位相同（`[0.42,-0.13,0.27]`） |
| C 不被重采样 | 15 s / 750 步命令零偏差，定时重采样 **0** 次 |
| D 回合边界隔离 | 7 个子条件全真：识别复位、命令清零、`trajectory_valid_mask=False`、内部步号归零、下一记录步号 0、回合计数 +1 |
| E 关节映射 | 12 关节映射 `[3,0,9,6,4,1,10,7,5,2,11,8]` 是排列；仿真按类型分组、SDK 按腿分组，**必须重排** |
| F 限幅与非法输入 | `[2.0,-2.0,5.0]→[1.0,-0.4,1.0]` 且 `clipped=True`；NaN→0 且 `invalid=True` |

### 28.3 本轮测试发现的四个真实缺陷（已修）

测试的价值就在这里——以下都不是文档问题，是代码行为错误：

**(1) 分段程序整体晚一个采样点。**
`torch.bucketize` 默认 `right=False`，边界值被归入**前**一段，于是 `t=1.0` 仍返回第 0 段。
`StepProgram`/`ComboProgram` 改为 `right=True`（段定义为 `[start, end)`）。
证据：TASK-018 的切换步由 `[51,101,151]` 变为 **`[50,100,150]`**，间隔精确 50 步 = 1.0 s。

**(2) warmup 边界多算一个样本。**
原来用 `ep_step*dt >= warmup_s`，把正好落在 1.0 s 的样本计入统计。改为严格 `>`，
"排除前 1.0 秒"才名副其实。

**(3) 终止原因取错行。**
原来读回合**最后一行**的 `reason_id`；若终止发生在中间，最后一行是 0（none），
`failed` 会被误判为 0。改为取**第一个** `terminated|truncated` 所在行的原因。

**(4) 步骤记录是 env 内部 buffer 的视图，会被下一步改写。**
`CommandStepRecord.terminated/truncated` 原先是 `torch.as_tensor(env 返回张量)`，
在 dtype/device 相同时不复制，**指向 termination manager 的缓冲区**。
调用方把 record 存下来，下一次 `step()` 就会把这些标志悄悄改掉。
症状很有迷惑性：同一次运行的调试打印显示 `trunc=True`，几行之后读同一个 record 变成 `False`。
修法：构造 record 时 `.clone()`。

### 28.4 回归验证（改动共享代码后必须重跑）

因为 (1)(2)(3)(4) 都在 TASK-018/020 的共享实现里，两项已验收任务全部重跑：

| 任务 | 结果 |
| --- | --- |
| TASK-018 九项契约 | **全 PASS**，`all_pass=true`；C 切换步变为 `[50,100,150]`（即缺陷 1 的修复生效），其余不变 |
| TASK-020 评估 + 复算 | 评估 `EVAL_EXIT=0`；独立复算 **204 项 0 失配，verdict=PASS** |

> 说明：缺陷 (2) 使 warmup 边界少算一个样本，因此 TASK-020 的数值与修复前有极小差异
> （相对量级 1e-4）。这不影响此前结论——复算与评估同源，`mismatches=0` 仍然成立。

### 28.5 用法

```bash
# 纯 CPU，随时可跑
python -m pytest tests/test_units.py -q          # 21 passed

# 需要仿真（tmux 内）
python scripts/verify_runtime_contracts.py --task CaMRA-Go2-Velocity --num_envs 4 --device cuda:0
```

### 28.6 已知边界

- 单元测试只覆盖纯逻辑；关节映射、限幅在真实 env 上的行为由运行时契约覆盖。
- 运行时契约用 4 个环境、单张 L40，验证的是**计数与相等关系**，不涉及性能。
- 本任务不改任何默认配置；`tests/` 与 `scripts/verify_runtime_contracts.py` 均为新增文件。

## 29. 短程训练与评估链路试运行（TASK-022）

**这不是正式训练，也不是最终测试。** 它的唯一目的是证明"训练 → 保存 → 恢复 → 评估"这条链路是通的，
并记录短训本身的不足。

### 29.1 先修掉两个挡路的问题

| 问题 | 现象 | 处理 |
| --- | --- | --- |
| 缺 `argcomplete` | `train.py` 导入即 `ModuleNotFoundError` | `pip install argcomplete`（只影响 CLI 补全，不改变训练语义） |
| 任务枚举过滤 | `--task CaMRA-Go2-Velocity` 报 *invalid choice*，列表里只有 `Unitree-*` | `train.py` 的过滤条件由 `"Unitree" in id` 放宽为 `("Unitree" in id or "CaMRA" in id)`。**没有改任务名**——改名会让 TASK-019～021 的工作记录失效 |

`train.py` 的这一行是本次唯一的上游脚本改动，原文件备份为 `train.py.bak-task022`。

### 29.2 短训结果（seed 0，120 次迭代，4096 环境）

```
Learning iteration 1/120    Mean reward: -0.65   Mean episode length: 18.18
Learning iteration 119/120  Mean reward: 15.24   Mean episode length: 982.96
TRAIN_EXIT=0
```

| 指标 | 起点 | 终点 |
| --- | --- | --- |
| Mean reward | −0.65 | **15.24** |
| Mean episode length | 18 步（0.36 s） | **983 步（19.7 s，接近 20 s 上限）** |

数值稳定：全程无 NaN、无 loss 爆炸、无环境重置异常。

**吞吐闭环**：`Total timesteps = 11,796,480`，纯训练约 420 s → **≈28,100 transitions/s**，
与 TASK-013 在 L40 上实测的 **27,970** 几乎一致。那条基准预测经受住了端到端（含 PPO 更新）的检验。

### 29.3 检查点与归属校验

产出 `model_0.pt / model_100.pt / model_119.pt`（各 4.5 MB），`params/{env,agent}.yaml`，`git/`，tfevents。

`tools/t022_check_ckpt.py` 做了 7 项检查，全部通过：

| 检查 | 实测 |
| --- | --- |
| env.yaml 是 Go2 | `usd_path=/pub/data/caohy/unitree_model/Go2/usd/go2.usd` |
| 速率 | `sim.dt=0.005`、`decimation=4` → 策略步 0.02 |
| 命令域 | `lin_vel_x=(-1.0,1.0) lin_vel_y=(-0.4,0.4) ang_vel_z=(-1.0,1.0)` |
| debug_vis | `False`（TASK-019 的覆盖生效） |
| 网络形状 | actor `[512,45] → [12,128]`、critic `[512,60]`，三个文件一致 |
| **不是 HiRA 的模型** | 未发现任何 48 维输入的策略 |
| agent.yaml | `camra_go2_locomotion`、seed 0、迭代 120（短训预算，非 3000 上限） |

`45/12/60` 三个维度同时对上，等于同时排除了"加载了 HiRA（48 维观测）"和"加载了别的机器人（不同动作维度）"。

### 29.4 显式恢复

从 `model_100.pt` 续训 110 次（目标 210）：

```
Learning iteration 132/210   Mean reward 15.03
...
Learning iteration 207/210
RESUME_EXIT=0
```

权重确实被加载：reward 从 15 量级继续（全新初始化是 −0.65 起），且**课程继续推进**：

| 阶段 | `lin_vel_cmd_levels` |
| --- | --- |
| 短训 120 次迭代全程 | **0.1000（从未推进）** |
| 恢复后续训至 209 | 0.1000 → 0.1375 → 0.2000 → 0.2708 → **0.3000** |

reward 也从 0.11 爬到 23.75（0.11 是新日志文件里第一个统计窗口，不是从零学习）。

⚠️ **一个需要记住的行为**：`--resume` 会在 `logs/rsl_rl/<exp>/` 下**新建一个时间戳目录**，
而不是续写原来的 run 目录。所以恢复后的产物（`model_200/209.pt`）与原 run 不在同一个文件夹，
追溯时必须靠 `--load_run <原目录名>` 这个参数记录。这是上游行为，不是本次引入的。

### 29.5 评估入口（开发验证集）

用 `eval_locomotion.py --checkpoint` 跑 4 个场景 × 16 回合（64 环境，seed 200，nominal 组）。
`run_info.json` 正确记录 `policy_source=checkpoint` 与 `checkpoint_sha256=a36aa6d8…`。

两个模型形成天然对照：

| 场景 | model_119（课程 0.10） | model_209（课程 0.30） |
| --- | --- | --- |
| stand | 失败 0.00，**pass** | 失败 0.00，**pass** |
| fwd_mid（0.6 m/s） | 失败 **1.00**，回合仅 1.50 s | 失败 **0.00**，回合跑满 10.00 s |
| start_stop | 失败 1.00 | 失败 0.00 |
| ramp | 失败 1.00 | 失败 0.00 |

失败率从 1.00 降到 0.00，说明评估链路对策略质量**敏感**，不是恒真或恒假。

但两者 `pass` 都不成立（除 stand），原因不是工具，而是：

**训练时的命令课程停在了 0.30，评估却要求 0.6。**

`fwd_mid` 的 rmse 反而从 0.383 升到 0.601 —— 因为 model_119 只撑 1.5 秒就摔（速度还没起来，误差小），
model_209 能站满 10 秒但面对 0.6 的命令只能跑出约 0 m/s（它只在 ≤0.3 的命令上学过），
于是误差稳定在 0.6 附近。**"不摔但追不上"比"摔得快"在 rmse 上更难看**，这是指标组合（失败率 + 精度）
必须一起看的原因。

### 29.6 记录的不足（本任务明确要求）

1. **120 次迭代远不够**。命令课程在短训里一次都没推进（0.1 全程不变），因为推进需要 tracking reward 达到阈值，
   这需要更多迭代。正式训练（3000 次）才会让命令域爬到 ±1.0。
2. 因此本任务的所有评估数值**不能**作为性能结论，只能证明工具与链路可用。
3. `resume` 新建目录的行为会让"同一条训练曲线"分散在多个文件夹，TASK-023 需要在 work record 里
   显式登记 run 目录的父子关系。

### 29.7 未解释的错误

无。剩下的输出都是已知且无害的：Blackwell 卡（sm_120）在 PyTorch 枚举所有 GPU 时的兼容性**警告**
（实际计算在 L40 上）、rsl_rl 关于 `obs_groups` 缺 key 的 fallback 警告（行为已正确，仅提示未来版本会移除）。

### 29.8 复现命令

```bash
# 短训（120 次）
bash /pub/data/caohy/scripts/t022_train.sh 120
# 显式恢复（从 model_100 续到 210）
bash /pub/data/caohy/scripts/t022_resume.sh <run_dir_name>
# 检查点归属
python tools/t022_check_ckpt.py <run_dir>
# 评估
bash /pub/data/caohy/scripts/t022_eval.sh <ckpt> 64 16 eval_dev
```

## 30. 阶段一交付与接手须知（TASK-023–TASK-028）

§23–§29 记录的是怎么走到这里；本章只写**现在有什么、怎么用、别踩什么坑**。
写这一章的原则：每条命令都在 tiger12 上真跑过，每个数字都能追到文件，做不到的事明说做不到。

### 30.1 交付物与校验入口

| 产物 | 位置 | 怎么校验 |
| --- | --- | --- |
| 第 4 轮正式训练 run | `/pub/data/caohy/logs/rsl_rl/rsl_rl/camra_go2_locomotion/2026-11-02_11-14-49_formal_s0_3000it_r4` | 同目录 `params/{agent,env}.yaml`、`git/` 由 train.py 落盘 |
| 选中检查点 | 同上 `model_2999.pt`（4,639,669 B） | sha256 `1c8f5aad38c2b763c7bceeda66d9a8f9e95eb463a358d768001868430f95efc9` |
| 导出候选包 | `/pub/data/caohy/camra_bundles/candidates/r4_it2999_v1` | `params/export_manifest.json` + `scripts/verify_export_consistency.py` |
| 最终测试（单次，不可重跑） | `/pub/data/caohy/logs/camra/TASK-024/{nominal,perturbed}/s2NN/` | 每种子 `summary.json`、`per_episode.csv`；根目录 `checkpoint.sha256` |
| 闭环重载 | `/pub/data/caohy/logs/camra/TASK-026/rollout/` | `t026_aggregate.json` 的 `verdict.pass` |
| **冻结包 v2（交付件）** | `/pub/data/caohy/camra_bundles/frozen/camra_go2_velocity_it2999_v2` | `frozen_manifest.json` + `.sha256`；`scripts/verify_frozen_bundle.py` |
| 冻结清单（仓库内副本，17 KB） | `CaMRA/locomotion/docs/frozen/camra_go2_velocity_it2999_v2.frozen_manifest.json` | 与包内清单同为 `6da14ce2…` |
| 评估协议（训练前固定） | `CaMRA/locomotion/docs/eval_protocol.json` | `status=fixed_before_training`，`user_approved=2026-10-31` |
| 计划与全部工作记录 | `doc/plan/20261031_camra_locomotion_implementation.json` | `validate_worklog.py … --mode update` |

结论一句话：final_test 种子 200–209 单次执行，nominal 10/10、perturbed 10/10 场景通过，
全局失败率 0.000 / 0.007；导出后闭环重载 JIT 与训练器逐位一致、ONNX 偏差 <9e-4；
冻结包全树 0444/0555 且校验 PASS。**这些数字都能从上表文件重新算出来，不依赖任何人的记忆。**

### 30.2 设备与环境

执行主机 tiger12（192.168.20.12），解释器 `/pub/data/caohy/miniconda/envs/env_isaaclab/bin/python`：

| 依赖 | 版本 |
| --- | --- |
| Python | 3.11.18 |
| torch | 2.7.0（CUDA 12.6） |
| numpy | 1.26.0 |
| isaaclab | 0.47.2 |
| isaacsim | 5.1.0.0 |
| rsl-rl-lib | 3.0.1 |
| onnxruntime | 1.31.1 |
| onnx | 1.23.2 |

GPU 用 **GPU0**（`CUDA_VISIBLE_DEVICES=0`）。两条实测教训：GPU2 长期被他人的 vLLM 占用，
占用后仿真会报 `Failed to get DOF velocities from backend`；本机 torch 构建在 GPU3/GPU4
（不同型号）上直接报 `CUDA error: no kernel image is available`。

所有命令必须在 tmux 内执行（计划顶层 `tmux_execution_policy`）：

```bash
tmux attach -t camra-locomotion     # 32 个窗口，按任务编号命名（TASK-023-*/T025*/T026RUN*/T027*）
tmux list-windows -t camra-locomotion
```

断线后先看窗口里是否已有进程在跑，**不要重复启动**；重跑会覆盖产物或与正在跑的进程抢 GPU。

### 30.3 真实命令

以下每条都在 tiger12 跑过，日志位置随命令给出。

**训练（第 4 轮配置，正式 3000 迭代，seed 0）**

```bash
bash /pub/data/caohy/scripts/t023_train_r4.sh 0 3000 formal_s0_3000it_r4
# 日志 /pub/data/caohy/logs/camra/TASK-023/formal_s0_3000it_r4.log
# 落盘 /pub/data/caohy/logs/rsl_rl/rsl_rl/camra_go2_locomotion/<时间戳>_formal_s0_3000it_r4/
```

**恢复训练（从中途检查点续跑，`--resume` 必须显式给 `--load_run` 与 `--checkpoint`）**

```bash
cd /pub/data/caohy/run/camra_go2
/pub/data/caohy/miniconda/envs/env_isaaclab/bin/python -u \
  /home/caohy/repositories/Paper/CaMRA/locomotion/scripts/rsl_rl/train.py \
  --task CaMRA-Go2-Velocity --headless --seed 0 --resume \
  --load_run 2026-11-02_11-14-49_formal_s0_3000it_r4 --checkpoint model_2000.pt \
  --max_iterations 3000 --device cuda:0
# 注意：resume 会新建一个 run 目录，同一条曲线会散在多个目录里，登记父子关系后再引用
```

**开发集选模（只在 dev 种子 100–109 上做，final_test 不参与）**

```bash
bash /pub/data/caohy/scripts/t023_finalize_r4.sh <run_dir>     # devsweep + t023_select_r4.py
bash /pub/data/caohy/scripts/t023_devfull_r4.sh <run_dir> 2999 # 10 个 dev 种子 × 30 回合确认
# 结果 /pub/data/caohy/logs/camra/TASK-023/（selection*_r4.json、dev_it2999*）
```

**最终测试（单次，跑完不得因结果重跑）**

```bash
bash /pub/data/caohy/scripts/t024_finaltest.sh \
  /pub/data/caohy/logs/rsl_rl/rsl_rl/camra_go2_locomotion/2026-11-02_11-14-49_formal_s0_3000it_r4/model_2999.pt
# 输出 /pub/data/caohy/logs/camra/TASK-024/{nominal,perturbed}/s2NN/summary.json
```

**导出 + 数值验证**

```bash
bash /pub/data/caohy/scripts/t025_export.sh          # 内含 export_policy_candidate.py 与 verify_export_consistency.py
# 或分步：
python scripts/export_policy_candidate.py --task CaMRA-Go2-Velocity \
  --checkpoint <ckpt> --out_dir /pub/data/caohy/camra_bundles/candidates/<tag> \
  --num_envs 64 --seed 100 --steps 300
python scripts/verify_export_consistency.py --bundle /pub/data/caohy/camra_bundles/candidates/<tag>
```

**闭环重载 + 三臂聚合判定**

```bash
bash /pub/data/caohy/scripts/t026_rollout.sh         # jit / onnx / trainer 各一个独立进程
python scripts/aggregate_rollout_arms.py --out_dir /pub/data/caohy/logs/camra/TASK-026/rollout
```

**冻结（版本号一旦用过不得复用；工作区不干净会被拒）**

```bash
python scripts/freeze_controller_bundle.py \
  --candidate /pub/data/caohy/camra_bundles/candidates/r4_it2999_v1 \
  --run-dir <run_dir> --checkpoint <run_dir>/model_2999.pt \
  --asset-root /pub/data/caohy/unitree_model/Go2 \
  --protocol docs/eval_protocol.json \
  --task024-root /pub/data/caohy/logs/camra/TASK-024 \
  --rollout-dir /pub/data/caohy/logs/camra/TASK-026/rollout \
  --t024-agg /pub/data/caohy/tools/t024_agg.py \
  --repo /home/caohy/repositories/Paper \
  --out-root /pub/data/caohy/camra_bundles/frozen --version <新版本名>
```

**校验冻结包（换机器也能跑，只需要 torch + onnxruntime + numpy）**

```bash
python scripts/verify_frozen_bundle.py \
  --bundle /pub/data/caohy/camra_bundles/frozen/camra_go2_velocity_it2999_v2 --samples 512
# PASS 时应看到 "problems": [] 与 [VERIFY-FROZEN] PASS
```

**不需要 Isaac 的数值回归（纯 CPU，可直接对只读冻结包跑）**

```bash
cd /tmp        # 冻结包是只读的，报告会写到当前目录而不是包内
python /home/caohy/repositories/Paper/CaMRA/locomotion/scripts/verify_export_consistency.py \
  --bundle /pub/data/caohy/camra_bundles/frozen/camra_go2_velocity_it2999_v2
# 19,200 条样本：jit 1.192e-06、onnx 1.118e-06、onnx-vs-jit 7.15e-07、确定性 0.0，PASS
# 包只读时脚本会明确打印 "manifest not updated"，不改动冻结包
```

**单元测试与运行时契约**

```bash
python -m pytest tests/test_units.py -q                    # 纯 CPU，当前 34 passed（约 2 s）
python scripts/verify_runtime_contracts.py                 # 需要仿真，tmux 内
python scripts/external_command_drive.py                   # 10 Hz 外部命令接口的独立驱动示例
```

**校准入口（下游，尚未实现）**

`CaMRA/calibration/` 目前只有用途说明。校准需要的输入已经齐备：冻结包的观测/动作契约
（`params/deploy.yaml` + `export_manifest.json`）、19,200 条真实观测样本、以及闭环重载脚本。
真正的标定要做的是：用同一冻结控制器在同一命令程序下采名义轨迹与执行轨迹、配对后拟合
成本裕度 α 与几何裕度 δ，并用独立数据检验——这一步**本次没有做**。

### 30.4 后续校准必须保持一致的五件事

任何一项变了，冻结包就不再代表现在这套行为，必须重新导出、重新做闭环判定、重新标定：

1. **模型**：`model_2999.pt` 的 sha256 `1c8f5aad…`。换检查点就是换策略，不是"微调参数"。
2. **命令域**：±1.0 m/s（vx）、±0.4 m/s（vy）、±1.0 rad/s（wz），采样区间
   `resampling_time_range=[4.0, 8.0] s`，外部命令 10 Hz 零阶保持。超出训练域的速度不给任何保证。
3. **频率**：sim 200 Hz（`sim.dt=0.005`）× decimation 4 → 策略 50 Hz（`step_dt=0.02`）。
   部署端若按别的频率调用，策略看到的是完全不同的动力学。
4. **限幅与滤波**：动作项 `JointPositionAction`，`q_target = default_joint_pos + 0.25·a`，
   wrapper `clip_actions=null`（term 的 ±100 实际不生效）；kp 25.0 / kd 0.5 / friction 0.01。
   要在部署端加自己的限幅或滤波，就是新增一层，必须重新评估。
5. **资产**：Go2 USD，资产树摘要 `95605072…`（已整体复制进冻结包 `assets/Go2/`）。
   换 URDF/USD 的惯量或关节限位，等于换机器人。

### 30.5 本次明确没做的

清单与冻结包 `frozen_manifest.json` 的 `not_done` 一致，不重复美化：

- **真机 `joint_ids_map` 重映射**：部署端 `State_RLBase.cpp` 按目标机器关节索引重排，本包给的是仿真侧顺序。
- **0–40 ms 指令延迟**：`eval_protocol.json` 的 `not_covered` 项，未建模、未验证。
- **sim2real**：状态估计噪声、执行器时延与摩擦差异、地形与材质多样性全部未覆盖。
- **ONNX batch=1**：图固定 `[1,45]→[1,12]`，只能逐帧推理；批推理需重新导出并重跑等效判定。
- **推理延迟不是实时性结论**：jit 中位 0.22–0.37 ms、onnx 1.18–1.24 ms 只是本机 headless 仿真的计算开销。
- **上层导航与校准模块**：本阶段只交付底层控制器。

### 30.6 给接手者踩过的坑

1. **Isaac 入口在未捕获 traceback 后仍返回 0**。所有批量脚本都按产物判定成功，而不是看退出码。
2. **`torch.inference_mode()` 会把仿真缓冲变成 inference tensor**，下一场景 reset 的原地写入直接崩
   （`Inplace update to inference tensor outside InferenceMode is not allowed`）。rollout 全用 `torch.no_grad()`。
3. **`ExternalCommandWrapper.reset()` 先让内部 env 算观测、之后才写程序指令**，所以 `reset()` 返回的那一拍
   观测携带的是复位时采样的随机指令。自下一拍起才一致。部署端自己复位时必须复现同一顺序。
4. **接触型动力学的混沌放大**：ONNX 与训练器每帧差 5e-7，7–45 个控制周期后就离开噪声地板，
   末位 xy 差可达 0.11 m，但区间统计量差 <1e-3。别拿逐帧轨迹对比判"导出坏了"。
5. **本仓库 `git status` 会报 stat 假阳性**（`<text: auto>` + CRLF）：`M` 但 `git diff --quiet HEAD` 为空。
   判脏必须按内容复核，并额外包含未跟踪文件。
6. **`AppLauncher` 自带 `--device`**，自定义同名参数会 `ValueError`；`parse_env_cfg` 要从
   `unitree_rl_lab.utils.parser_cfg` 导入，别从 `isaaclab_tasks.utils`。
7. **rsl-rl ≥3 的 policy 按观测组名索引**，传扁平张量会 `IndexError`，要传 `TensorDict`/dict。
8. **ONNX 不支持一次喂 19,200 条**（图 batch 固定 1），验证脚本一律逐条循环。
9. **train.py 的 `--resume` 新建 run 目录**，同一曲线会散开；引用检查点时带上完整路径与 sha256。
10. **`tmux new-window -d` 后必须再 `send-keys` 才会真正执行**；只看会话存在不算跑起来。

### 30.7 本章命令的实际运行证据

写文档的同时把不需要长时间的命令真跑了一遍，证据都在 `/pub/data/caohy/logs/camra/` 下：

| 命令 | 日志 | 结果 |
| --- | --- | --- |
| `pytest tests/test_units.py -q` | `TASK-028/doc_commands.log` | `34 passed, 3 warnings in 1.89s` |
| `aggregate_rollout_arms.py --out_dir TASK-026/rollout` | 同上 | `VERDICT: PASS`（三臂复位采样流一致 1.8291529417037964） |
| `verify_export_consistency.py --bundle <frozen v2>` | `TASK-028/verify_frozen_readonly.log` | 19,200 样本 PASS；jit 1.192e-06、onnx 1.118e-06、joint 2.794e-07、确定性 0.0 |
| `verify_frozen_bundle.py --bundle <frozen v2> --samples 512` | `TASK-027/verify_v2.log` | 29 文件 PASS，重放 jit 1.43e-06 / onnx 9.54e-07 |
| `t027_negative_v2.sh`（拒绝重出/拒绝写入/篡改必败/原包仍过） | `TASK-027/negative_v2b.log` | 4/4 符合预期 |

同一份 `doc_commands.log` 里还保留了修复前的失败：`verify_export_consistency.py` 原本把报告写回
`<bundle>/params/`，对只读冻结包直接 `PermissionError`。**校验工具不该要求被校验对象可写**，
已改为只读包自动写到当前目录并明确打印 `manifest not updated`；冻结包本身一个字节都没动。

## 31. 底层验收证据复核（TASK-029，只读）

本章只核对**已经存在的**训练、评估与冻结产物，不运行训练、不重跑评估、不修改任何被冻结文件。
结论按三类标注：**[直接验证]** 本次亲手比对过；**[历史报告]** 只有当时的记录可查；
**[缺证]** 找不到可核对的来源。核对脚本与日志见 `31.7`。

复核基线：`02f7c58`（计划 v3.1.0），冻结 v2 清单 `6da14ce2…`，评估协议 sha256
`0afc2cea…`。执行主机 tiger12（`caohy@192.168.20.12`），tmux 会话 `camra-locomotion`。

### 31.1 证据索引

| 证据 | 位置 | 本次核对方式 | 结论 |
| --- | --- | --- | --- |
| 选定检查点 `model_2999.pt` | `<run>2026-11-02_11-14-49_formal_s0_3000it_r4/` | 重量 sha256 | **[直接验证]** `1c8f5aad…`，4,639,669 B |
| 冻结包 v2（29 文件） | `/pub/data/caohy/camra_bundles/frozen/camra_go2_velocity_it2999_v2` | 全树重验 + 样本重放 | **[直接验证]** PASS |
| 包内检查点副本 | `model/model_2999.pt` | 与上游文件比散列 | **[直接验证]** 同一散列 `1c8f5aad…` |
| 导出图 JIT / ONNX | `exported/policy.pt` / `.onnx` | 重量 | **[直接验证]** `8e3f14ad…` / `d0c993b6…` |
| 评估协议版本 | `docs/eval_protocol.json` | 与 20 个 `run_info.json` 的 `protocol_sha256` 比对 | **[直接验证]** 全部 `0afc2cea…`，v1.0.0 |
| 产生结果的评估代码 | `scripts/eval_locomotion.py` | `git diff 740224c..HEAD` | **[直接验证]** 与当时逐字相同 |
| 最终测试（20 目录 / 6000 回合） | `TASK-024/{nominal,perturbed}/s2xx/` | 逐目录 `run_info` + `per_episode.csv` | **[直接验证]** 见 31.2 |
| 开发集确认（10 种子） | `TASK-023/dev_it2999_s1xx_r4/` | 原始步文件 | **[直接验证]** 见 31.3 |
| 闭环重载三臂 | `TASK-026/rollout/*.json` | 文件存在 + 汇总判定 | **[直接验证]** 4 个 JSON + 聚合 PASS |
| 训练资源与环境 | `env/pip-freeze.txt`、`params/*.yaml` 在包内 | 存在性 | **[直接验证]** |
| 训练当时的 GPU/时长/排队 | — | 无独立记录 | **[缺证]** 只有 session 时间戳 |
| 是否存在 seed=1/2 的训练授权 | 计划、交接笔记 | 全文检索 | **[缺证]** 无授权记录，run 树也无对应目录 |

### 31.2 训练与模型数量：**实际只有一个训练模型**

15 个 run 目录的 `params/agent.yaml` **全部** `seed: 0`：

```
2026-10-31_22-35-31_formal_s0_3000it        seed:0  envs:4096  ckpts=31 last=2999
2026-11-01_09-41-35_formal_s0_3000it_r1     seed:0  envs:4096  ckpts=31 last=2999
2026-11-01_16-12-39_formal_s0_3000it_r2     seed:0  envs:4096  ckpts=11 last=1000   ← 中止
2026-11-01_18-39-18_formal_s0_3000it_r3     seed:0  envs:4096  ckpts=31 last=2999
2026-11-02_11-14-49_formal_s0_3000it_r4     seed:0  envs:4096  ckpts=31 last=2999   ← 交付这一个
```

五次 formal 是**同一 seed 下的调参重跑**（奖励与课程不同），不是五次独立训练样本。
**[直接验证]** 因此：交付的模型数 = 1，而协议 `trial_budget` 写的是三个训练种子 —— 二者不一致。
**[直接验证]** 十个最终测试种子是**评估种子**，与训练模型数无关；`run_info.env_seed` 分别为
200–209，不能当作十个模型。这一点在 TASK-024 的记录里没有混淆，但协议措辞容易被误读。

### 31.3 场景变体覆盖：**多方向场景实际只跑了一个方向**

协议里有三个多方向场景：`lateral`（±0.3 m/s，2 个变体）、`turn_inplace`（±0.8 rad/s，2 个）、
`combined`（4 个符号组合）。从**归档的原始步文件**统计每回合实际下发的指令：

| 数据集 | lateral | turn_inplace | combined |
| --- | --- | --- | --- |
| 最终测试 nominal（300 回合/场景） | 300× `vy=+` | 300× `wz=+` | 300× `(+,+,+)` |
| 最终测试 perturbed（300 回合/场景） | 300× `vy=+` | 300× `wz=+` | 300× `(+,+,+)` |
| 开发集 seed 100–109（300 回合/场景） | 300× `vy=+` | 300× `wz=+` | 300× `(+,+,+)` |

三个数据集、两个分组、1800 个回合里，**负方向与其余三个组合一次都没跑过**：
`lateral` 覆盖 1/2，`turn_inplace` 覆盖 1/2，`combined` 覆盖 1/4。

**[直接验证]** 机制在代码里可定位：`eval_locomotion.py` 用 `ep_counter % len(variants)` 分配变体，
而 `ep_counter` 对所有环境从 0 起（`torch.zeros(bridge.num_envs)`），回合又都在同一仿真步因
`reached` 同时结束，`while len(done_rows) < wanted` 在收满 30 条时就退出 —— 于是第一批 64 个
环境全都拿的是变体 0，取前 30 条即为 30 个变体 0。`--num_envs 64` > `--episodes_per_scenario 30`
使这个缺口必然出现，与随机性无关。

**[直接验证]** 同一原因还钉死了环境身份：nominal 每个场景只用到 `env_id 0–29`（30/64），
perturbed 因偶发终止才用到 31 个（`max env_id 60`）。

**这不证明模型无效**：模型对同号指令的跟踪能力有实测；但「左右对称、四个组合全过」这句话
目前没有证据支撑，TASK-024 的 `overall_pass` 实际只覆盖了半个（组合场景四分之一）命令域。
修复与按补充协议复测分别属于 TASK-031 与 TASK-036。

### 31.4 指标完整性：汇总层不保留有效数

**[直接验证]** `summary.json` 里每个场景的 `episodes` 是**整数计数**，没有逐变体、逐指标的
有效数；判定只看聚合值与阈值。逐回合明细在 `per_episode.csv`，可复核，但没有任何环节要求
「有效样本数 ≥ 某个下限」。

本次从 CSV 统计的有效数（`TASK-024`，每场景 300 回合/组）：

* 绝大多数指标 300/300 有效；
* **`fwd_low` 的 `rise_time_s` 在 nominal 只有 86/300 有效**（perturbed 为 252/300）——
  该场景的阈值不含 `rise_time_s`，所以没有影响通过判定，但它实证了「指标级有效数可以远低于
  回合数，而汇总层不会察觉」。

**[直接验证]** 终止原因：nominal 全部 `scenario_end`；perturbed 共 12 次 `bad_orientation`
（fwd_low 1、fwd_mid 1、fwd_high 2、backward 1、lateral 1、turn_inplace 1、combined 1、
start_stop 2、ramp 2，共 3000 回合中的 12 次），均计入 `failure_rate`，未被静默丢弃。

### 31.5 分类结论

**已直接验证**

* 检查点 → 冻结包 → 导出图 的散列链完整，包内副本与上游文件同散列；
* 冻结包 29 文件全树重验 PASS，重放样本 JIT 1.43e-06 / ONNX 9.54e-07；
* 20 个最终测试目录的协议散列、代码提交、`num_envs`、`episodes_per_scenario`、`step_dt` 完全一致；
* `group_applied` 与分组一致：nominal 为恒等扰动，perturbed 为摩擦 `[0.3,1.25]`、质量 `[-1,3] kg`、
  推力 `[-0.5,0.5] m/s`、观测噪声按训练值开启；
* 每场景 300 回合/组，终止原因可追溯；
* 多方向场景只覆盖变体 0（见 31.3）。

**仅历史报告**

* TASK-024 的「1h57m44s、20 次评估 exit=0」只有当时日志的时间戳；
* 训练资源占用、排队与总机时没有独立记录；
* TASK-026 三臂闭环的数值结论来自当时汇总，本次只确认产物文件存在与聚合判定可复跑。

**缺证待核**

* 三训练种子要求与实际单模型的差异，**没有任何授权记录**说明曾被批准改为单模型 —— 这属于
  TASK-030 必须由用户裁决的决策，本任务只报告事实；
* 训练当时的 GPU 型号/占用（当前 nvidia-smi 已被他人任务占用，不能回推当时状态）。

### 31.6 本任务没有做的事

未运行训练、未重跑评估、未修改冻结包、未改写 TASK-001～028 的任何记录。评估工具的缺陷
（变体分配、有效数判定）只记录、不在此修复 —— 修复属于 TASK-031 与 TASK-032，复测属于 TASK-036。

### 31.7 复核脚本与日志

| 脚本 | 产出 | 用途 |
| --- | --- | --- |
| `t029_audit1.sh` | `TASK-029/audit_env.log` | 主机、Git、tmux、进程、run 目录盘点 |
| `t029_variants2.py` | `TASK-029/variant_coverage2.log` | 从原始步文件统计实际变体 |
| `t029_metric_validity.py` | `TASK-029/metric_validity.json` + `.log` | 逐指标有效数、终止原因、环境身份 |
| `t029_runinfo.py` | 控制台 | 20 个目录的一致性 |
| `t029_hash2.py` | 控制台 | 检查点与导出图散列 |

以上均只读；脚本本体在工作区 `_t029/`，运行副本在 `/pub/data/caohy/tools/`，
日志在 `/pub/data/caohy/logs/camra/TASK-029/`。tmux 窗口：`T029A1B`（环境）、`T029V2`（变体）、
`T029M`（指标）、`T029E`（散列与运行清单）。

### 31.8 后续任务可直接使用的路径

| 用途 | 路径 | 本次状态 |
| --- | --- | --- |
| 选定模型 | `<run>/model_2999.pt`，`<run>=/pub/data/caohy/logs/rsl_rl/rsl_rl/camra_go2_locomotion/2026-11-02_11-14-49_formal_s0_3000it_r4` | 存在，散列已核 |
| 训练运行参数 | `<run>/params/{agent,env,deploy}.yaml`、`camra_env_cfg.py` | 存在，与冻结包一致 |
| 冻结包 | `/pub/data/caohy/camra_bundles/frozen/camra_go2_velocity_it2999_v2` | 只读，复验 PASS |
| 资产 | `/pub/data/caohy/unitree_model/Go2`（`UNITREE_MODEL_DIR`） | 存在 |
| 评估协议 | `CaMRA/locomotion/docs/eval_protocol.json`（`0afc2cea…`） | 与当时一致 |
| 最终测试原始数据 | `/pub/data/caohy/logs/camra/TASK-024/{nominal,perturbed}/s2xx/` | 完整，可复算 |
| 开发集原始数据 | `/pub/data/caohy/logs/camra/TASK-023/dev_it2999_s1xx_r4/` | 完整，可复算 |
| 闭环重载 | `/pub/data/caohy/logs/camra/TASK-026/rollout/` | 4 个 JSON + 聚合 |
| 复核证据（本次） | `/pub/data/caohy/logs/camra/TASK-029/` | 新增 |

**尚缺、需 TASK-034 处理的**：TASK-024/023/026 的驱动脚本（`t023_devfull_r4.sh`、`t024_finaltest.sh`、
`t024_agg.py` 等）目前只在 `/pub/data/caohy/scripts|tools/` 下，**不在仓库内**；模型与原始日志也不进 Git。
换设备仅凭 clone 仍不能复现，这一点已写入 TASK-034 的目标。

## 32. 变体覆盖修复（TASK-031）

上一轮复核（第 31 章）发现多方向场景只跑了变体 0。本章记录修复与证据。
模型、命令域、控制频率、奖励、协议阈值、冻结包**均未改动**，改动只在评估工具的调度与记录。

### 32.1 缺陷：四条独立原因，缺一条就不成立

1. **变体选择用了从 0 开始的计数器**：`sel = (ep_counter % n) == vi`，场景开始时
   `ep_counter` 全为 0，所以第一个仿真步 64 个环境**全部**在跑变体 0。
2. **终止条件只看场景总数**：`while len(done_rows) < wanted`，而 `num_envs=64 > wanted=30`，
   第一批 30 条收满即退出，环境根本没机会跑第二个回合。
3. **没有任何字段记录「这一回合属于哪个变体」**：事后无法从 `per_episode.csv` 或 `summary` 反查，
   漏测在报告里表现为“通过”。
4. **步数保护命中后照常判定**：`guard` 超限只打印一行 WARN 就 `break`，
   随后用不足的回合数聚合，`overall_pass` 仍可能为真。

### 32.2 修复

* 新增 `source/unitree_rl_lab/unitree_rl_lab/utils/variant_scheduler.py`（纯 Python，无 Isaac）：
  - `assign_variants(num_envs, n)` 固定槽位 `i % n`，各变体分到的环境数相差不超过 1；
  - `VariantScheduler` 按变体独立配额：`record(env_id)` 返回 `{variant, episode_id, accepted}`，
    超额回合计入 `rejected`（不静默丢弃），`coverage()` 输出逐变体接受/拒绝/缺失/未覆盖；
  - `num_envs < num_variants` 直接抛错——这种配置下必有变体永远跑不到。
* `scripts/eval_locomotion.py`：
  - 配额改成**逐变体**：新增 `--episodes_per_variant`（协议 D-2 用 30/变体/种子）；
    `--episodes_per_scenario` 保留兼容，缺省时按 `// n_variants` 折算并打印一行说明；
  - 命令按固定槽位下发，循环条件是 `until sched.complete()`；
  - `per_episode.csv` 新增 `variant_id`、`episode_id`、`env_seed`、`cmd_vx_first/cmd_vy_first/cmd_wz_first`；
  - `raw/<group>/<scenario>.npz` 新增 `variant_ids`、`episode_ids`（与 `lengths` 等长）；
  - `summary.json` 改为 `scenarios[].variants[]`（逐变体判定）+ `coverage` + `combined`（仅对照）
    + `incomplete_reason`；场景通过 = 覆盖率完整且**所有变体**通过；`overall_pass` 再叠加
    `coverage_complete`（协议 D-4）。

### 32.3 证据

| 项 | 结果 |
| --- | --- |
| 新增单测 | `tests/test_variant_scheduler.py` 13 项，含“旧分配器只跑变体 0”的复现用例 |
| 全量单测 | `python -m pytest tests/test_variant_scheduler.py tests/test_units.py -q` → **47 passed** |
| 小批真跑 | tmux `camra-locomotion:T031A`，GPU 0，seed 300，quota 4/变体，nominal：`lateral` accepted `[4,4]`、`combined` `[4,4,4,4]`，`coverage_complete=True` |
| 命令级核对 | 从 `raw/nominal/*.npz` 首步命令读回：lateral v0 `(0,+0.3,0)` / v1 `(0,−0.3,0)`；combined 四变体 `(0.6,0.3,0.5)`、`(0.6,−0.3,−0.5)`、`(−0.6,0.3,−0.5)`、`(−0.6,−0.3,0.5)` |
| 步数保护 | tmux `T031B`，quota 1000/变体：4201 步命中上限，`accepted=[256,256]`、`missing={0:744,1:744}`、`incomplete_reason=step_guard_hit`、`overall_pass=False` |

产物：`/pub/data/caohy/logs/camra/TASK-031/{smoke_v1,smoke_v2,smoke_guard}/`，日志同名 `.log`。
注意 Kit 的 C++ 启动横幅在退出时才 flush，**日志文件末尾不是运行结果**，结果看 `[EVAL]`/`[EVAL_SUMMARY]` 行或 pane。

### 32.4 顺带暴露的一个真实问题（仅示意）

小批 4 回合/变体的 `lateral`：**右移 v0 的 `rmse_vy=0.0886` 过了 0.10 阈值，左移 v1 的 `0.1037` 没过**。
旧口径只跑右移，这类左右不对称永远不会出现在报告里。样本只有 4 回合，不能当结论，
但足以说明复测（TASK-036）必须逐变体看，不能合并平均。

### 32.5 兼容性变化与没做的事

* `summary.json` 结构变了（`scenarios[]` 下新增 `variants`/`coverage`/`combined`/`incomplete_reason`，
  `pass` 改为逐变体判定）。读旧字段的脚本要同步改——`TASK-034` 把驱动脚本入库时一并处理。
* 退出码仍为 0（“跑完了”= 0），判定只看 `summary.json`；是否“不完整即非零”留给 `TASK-032` 决定。
* `aggregate()` 对非有限值（如 `rise_time_s` 的 NaN）仍然丢弃，未改——属 `TASK-032`。
* 未跑正式复测，未改模型/阈值/冻结包，未重写 TASK-023/024 的任何记录。

## 33. 有效数与完整性判定（TASK-032）

第 31 章记录的现象是「汇总层不保留有效数」：`aggregate()` 把非有限值过滤掉，再对剩下的求均值，
于是「1 个有效值 + 29 个缺失」与「30 个有效值」在报告里长得一样，都能 PASS。本章修复整条判定链。
阈值、模型、冻结包均未改动。

### 33.1 缺陷清单（逐条都能单独造成假 PASS）

1. `aggregate()` 用 `vals[np.isfinite(vals)]` 过滤后求均值，`valid_count` 从不记录 → 缺失不可见。
2. `compute_metrics()` 对 `rise_time_s`/`stop_drift_m`/`overshoot_mps` 直接返回 `nan`，**不说明原因**，
   无法区分「程序没定义这个指标」与「机器人没响应」。
3. `warmup` 后样本不足 2 个时，代码静默改用全部样本（包括 settle 段）继续算，等于把预热误差算进指标。
4. 空输入/缺键：`r[m]` 直接取值，缺键会抛 `KeyError`；`n=0` 时 `failure_rate` 为 nan，
   `bool(nan <= 0.05)` 为 False 只是碰巧挡住了。
5. **文本真值陷阱**：CSV 里 `failed` 是字符串，`bool(r["failed"])` 对 `"0"` 也返回 True。
   任何从 CSV 读回合再聚合的脚本（复算、跨种子汇总）都会把失败率算成 100%。
6. 缺场景/缺分组没有任何完整性概念：只聚合「有的东西」，剩下的静默为 0 条却仍输出一个总体结论。

### 33.2 修复

| 位置 | 改动 |
| --- | --- |
| `compute_metrics()` | 新增 `invalid_reasons`（逐指标给原因）、`warmup_short`、`warmup_samples`；`rise_time_s` 区分 `not_reached`（命令有速度但没到 90% 并保持 0.2 s）与 `undefined_program`（纯站立/纯转向命令没有前向阶跃）；`stop_drift_m` 增加 `window_truncated`；单步轨迹标 `short_trajectory` |
| `aggregate()` | 逐指标输出 `expected_count / valid_count / na_count / invalid_count / valid_ratio / invalid_reasons / status`；`status ∈ {pass, fail, insufficient_evidence, reported_only}`；新增 `complete` 与 `incomplete_reason`；默认 `MIN_VALID_RATIO=0.90`（协议 D-3） |
| 分母规则 | **只有** `undefined_program` 计入 N/A 并从分母剔除；`not_reached / window_truncated / short_trajectory / warmup_not_reached / missing_key / non_finite` 一律算缺失证据，进分母、拉低有效比例 |
| 文本解析 | `_as_bool()`/`_as_failed()`/`_reasons_of()`：`"False"` 不再为真、`"0"` 不再算失败、CSV 里的 JSON 原因串可解析 |
| 变体定义 | 变体符号组合收进 `eval_metrics.variant_signs()`，评估器与跨种子汇总共用，避免「几个变体」两边算得不一样 |
| `aggregate_eval_runs.py`（新） | 按 `<root>/<group>/s<seed>/per_episode.csv` 汇总：缺 run、缺场景、缺变体、配额不足都置 `complete=False`；退出码 0=通过 / 1=完整但失败 / 2=不完整 |
| `recompute_metrics.py` | 比对项增加 `variant_id`、`episode_id`、`invalid_reasons`（JSON 规范化比较），使「哪些指标不可测、为什么」也可复算 |

### 33.3 证据

* 单测 **71 passed**（`pytest tests/ -q`）：34 既有 + 13 调度 + 18 有效性 + 6 跨种子汇总。
  有效性用例直接构造「1 有效 + 29 缺失」「全部非有限」「缺键」「空输入」「warmup 不足」「单步轨迹」
  与「合法 N/A」两两对照，不靠被测函数自证。
* 真跑（tmux `camra-locomotion:T032A`，quota 4/变体，seed 300，combined+start_stop）：
  `combined` 四变体、`start_stop` 单变体全部 `coverage_complete=True`；`summary.json` 每个 check 都带
  `valid_count/expected_count`（如 `rise_time_s` 4/4）。
* 跨种子汇总（tmux `T032B`）：
  - `--seeds 300` → `runs 1/1` 但 `complete=False`、退出码 **2**：这次只跑了 10 个场景中的 2 个，
    其余 8 个没有回合，**不再输出看似完整的 PASS**；
  - `--seeds 300-301` → `runs 1/2`、`complete=False`、退出码 2，缺的 run 被列在 `missing_runs`。

### 33.4 判定三态

`pass`（达标）、`fail`（证据齐但超阈值）、`insufficient_evidence`（有效数不足，既不算过也不算失败）。
只有「完整 + 全部 pass」才会 `overall_pass=true`；证据不足单独计数，不混进失败率。

### 33.5 没做的事

* 未跑正式复测（TASK-036），未改阈值/模型/冻结包。
* `TASK-023/024` 的历史 `summary.json` 仍是旧结构，不重写；复测产物用新结构。
* 汇总工具只做「读 CSV → 判定」，不重算指标；指标复算仍由 `recompute_metrics.py` 负责。

## 34. 停车概率渐增的时间单位（TASK-033）

结论先写：配置里的 `stop_event_ramp_iters = 300` 是 **300 个控制步**，不是 300 个 PPO 迭代。
按实际配置折算为 **12.5 个迭代 / 6.0 s 仿真时间**。文档与注释此前把它写成「300 iterations」，
差 24 倍。本章只纠正文字与单位，**参数值一个都没改**，所以已交付模型的行为逐位不变。

### 34.1 调用链与单位（可复算）

| 环节 | 事实 | 证据 |
| --- | --- | --- |
| 计数器 | `env.common_step_counter` 在 `ManagerBasedRLEnv.step()` 里每次 `+1` | `/pub/data/caohy/IsaacLab/source/isaaclab/isaaclab/envs/manager_based_rl_env.py:203`（注释原文 “total step (common for all envs)”） |
| 一次 `step()` | `decimation=4`、`sim.dt=0.005` → 一个控制步 = 4 个物理步 = **0.02 s**，策略 50 Hz | `camra_env_cfg.py:172`；`run_info.json` 的 `step_dt=0.02` |
| 一个 PPO 迭代 | `num_steps_per_env = 24` 个控制步 = **0.48 s** 仿真 | `rsl_rl_ppo_cfg.py:12` |
| 渐增长度 | `ramped_probability(p_max, common_step_counter, 300)` | `velocity_command.py:30-33`、`stop_events.py:ramped_probability` |
| 折算 | 300 控制步 ÷ 24 = **12.5 迭代**；300 × 0.02 = **6.0 s** 仿真 | 上式直接代入 |

`ramped_probability` 在 `iteration >= ramp_iters` 时饱和到 `p_max=0.25`，因此
**第 13 个迭代起停车事件概率已经是满值**，而不是第 300 个迭代。

补充观察（不改结论）：round 4 冒烟日志里前 60 个迭代回合长度卡在 ~16 步，而 round 3 到第 90 个迭代
已恢复到 562 步——这与「渐增保护了早期阶段」的说法并不一致，因为概率在第 13 个迭代就已经满了。
早期停滞更可能来自停车事件本身而不是渐增长度。

### 34.2 改了什么（仅文字）

| 文件 | 原文 | 现在 |
| --- | --- | --- |
| `stop_events.py` | “Ramping in over ~300 iterations keeps the early phase identical to round 3…” / “warm-up length in iterations” | 去掉错误结论，补 “300 control steps = 12.5 iterations = 6.0 s”，参数说明改为 control steps |
| `camra_env_cfg.py`（override 说明） | “Ramp length 300 iterations … iterations 0-300 match round 3 exactly” | 改为 control steps，并写明只有最初约 6 s 的仿真经验与 round 3 不同；注明数值故意保持不变 |
| `camra_env_cfg.py`（行内注释） | “ramped in over the first 300 iterations so the early gait phase is untouched” | “300 CONTROL STEPS (= 12.5 PPO iterations …, = 6.0 s simulated), not 300 iterations” |
| `velocity_command.py` | “Iterations over which … is ramped in linearly” | “Control steps over which …”，并注明计数器是 `common_step_counter` |

### 34.3 行为不变的证据

* `git diff` 只包含注释、docstring 与一处测试；`stop_event_prob=0.25` 与
  `stop_event_ramp_iters=300` 两处赋值原样保留（`camra_env_cfg.py:190-193`）。
* 新增测试 `test_ramp_length_is_control_steps_not_ppo_iterations`：固定
  `300/24 = 12.5`、`300×0.02 = 6.0 s`、第 288 步概率 0.24、第 300 步饱和，防止文字再次漂移。
* 全量单测 **72 passed**（原 71 + 本任务 1）。

### 34.4 若真要「300 个迭代」的渐增

那需要把 `ramp_iters` 改成 `300 × num_steps_per_env = 7200`，这会改变前 300 个迭代的命令分布，
即改变训练任务本身，**不是文档修正**。本轮没有该授权，也不应该用现有模型去“验证”一个它没训练过的分布。
若要改，按计划另立训练与验证任务（重新训练 + 重新评估 + 重新冻结），`TASK-037` 已为此留了位置。

## 35. 仓库内复现入口与产物索引（TASK-034）

### 35.1 缺口的真实形状

README 里五条命令指向 `/pub/data/caohy/scripts/`（`t023_train_r4.sh`、`t023_finalize_r4.sh`、
`t024_finaltest.sh`、`t025_export.sh`、`t026_rollout.sh`），冻结命令还要
`--t024-agg /pub/data/caohy/tools/t024_agg.py`。这些都是本轮临时写的脚本，**不在仓库里**，
所以"clone 下来能不能跑"这个问题的答案是"不能"，而且失败点分散：解释器路径、日志根、资产根、
选模脚本、聚合脚本、检查点路径各写死一处。

`source/unitree_rl_lab/unitree_rl_lab/assets/robots/unitree.py` 第 20 行原本是

```python
UNITREE_MODEL_DIR = "/pub/data/caohy/unitree_model"  # Tiger12 本地资产
```

也就是说**连源码都绑定了这一台机器**。

### 35.2 现在有哪些仓库内入口

| 原临时脚本 | 仓库内入口 | 行为差异（都写进脚本注释） |
| --- | --- | --- |
| `t023_train_r4.sh` | `scripts/repro/train.sh [seed] [iters] [name]` | 路径来自配置；`--device` 取 `CAMRA_DEVICE`；provenance 记到 `$CAMRA_LOG_ROOT/repro/train/` |
| `t023_devsweep_r4.sh` | `scripts/repro/devsweep.sh <run_dir> [seed] [eps] [envs] [suffix] [iters...]` | suffix 改为参数（不再写死 `_r4`）；产物判据从"退出码"改为"summary.json 存在" |
| `t023_finalize_r4.sh` | `scripts/repro/finalize.sh <run_dir> [suffix]` | 返回值取第一个失败而不是两码相加（相加可能互相抵消成 0） |
| `t023_select_r4.py` | `scripts/repro/select_checkpoint.py` | 目录名按正则 `dev_it<数字><suffix>` 精确匹配（原实现用 glob，见 35.7）；同时报出 per-scenario 归一化的分数 |
| `t024_finaltest.sh` | `scripts/repro/finaltest.sh <ckpt> [group] [seeds] [eps] [envs]` | 支持单分组；非空输出目录需 `--allow-existing`；逐种子判定产物 |
| `t024_agg.py` | `scripts/aggregate_final_test.py` | 只是池化均值口径的复现器；新协议判定在 `aggregate_eval_runs.py` |
| `t025_export.sh` | `scripts/repro/export.sh <ckpt> [out_dir] [tag]` | 默认输出 `candidates/<run>_<iter>`；产物检查保留 |
| `t026_rollout.sh` | `scripts/repro/rollout.sh <bundle> <ckpt>` | 三臂跑完直接调 `aggregate_rollout_arms.py` 并返回它的退出码 |

所有 shell 入口共同 source `scripts/repro/_common.sh`：它从**自身位置**推仓库根（不看 `$PWD`），
读 `config/repro.env` 但**只填未被导出的变量**，然后设 Isaac 的 `XDG_*`、`TMPDIR`、`EULA`。

### 35.3 路径只有一处默认值

`config/repro.env` 是唯一默认来源，解析顺序是 **导出的环境变量 > 该文件 > 模块内默认值**，
`scripts/camra_config.py` 的 `cfg.describe()` 会打印每个值的来源（`env` / `file` / `default`），
预检输出开头就是这张表——出问题时先看它，比读代码快。

换机器只改一件事：

```bash
export CAMRA_ASSET_ROOT=/mnt/assets/unitree_model
export CAMRA_LOG_ROOT=/mnt/logs/camra
```

资产根有三处消费方，都读同一个变量：`unitree.py`（`UNITREE_MODEL_DIR` 优先、`CAMRA_ASSET_ROOT` 次之）、
`scripts/repro/*.sh`（存在性检查）、`freeze_controller_bundle.py`（`--asset-root` 默认值 = `$CAMRA_ASSET_ROOT/Go2`）。

### 35.4 入库边界与产物索引

代码入库，**模型、原始日志、冻结包不入库**。`docs/artifacts_index.json` 是这份边界的机器可读版本：

| 条目 | 散列 | 体量 |
| --- | --- | --- |
| `frozen_bundle_v2` | `8923c6fd4f05a921…`（树） | 29,127,046 B / 31 文件 |
| `frozen_bundle_v1_superseded` | `9cf9a07099ed195c…`（树） | 29,124,660 B / 31 文件 |
| `candidate_bundle` | `e3b21eb382a123d8…`（树） | 5,454,281 B / 6 文件 |
| `checkpoint_selected` | `1c8f5aad38c2b763…` | 4,639,669 B |
| `training_run_params` | `922d473cb449b2cb…` | 32,299 B |
| `assets_go2` | `bd2fcb2f235b979f…`（树） | 18,815,207 B / 4 文件 |
| `raw_logs_task024` | `8df2250af159c16a…`（树） | 662,001,271 B / 362 文件 |
| `raw_logs_task026` | `32ea7954eefb787f…`（树） | 10,597,117 B / 16 文件 |
| `protocol` / `protocol_supplement` | `0afc2cea…` / `f50d8b7f…` | 9,167 B / 6,640 B（入库） |

目录条目是树散列：对目录内每个文件算 sha256，再对排序后的 `相对路径:sha256` 列表取 sha256，
所以换内容、加删文件、改目录结构都会变。`verify` 按**当前解析到的路径**重算，只比对内容散列、
不比对路径字符串——换机器不会因为路径不同而误报，但少了文件一定报。

取用与换设备四步写在 `docs/reproduction.md`（rsync 命令、四步流程、明确没做的事）。

### 35.5 预检先分清"谁的错"

`scripts/check_repro_readiness.py` 的三种判定是本任务的核心产出：

| VERDICT | 退出码 | 含义 |
| --- | --- | --- |
| `ready` | 0 | 该层级需要的资源齐备，可以开跑 |
| `infra_blocked` | 2 | 检查点/资产/协议/历史日志缺失——**控制器根本没被运行，不能据此评价它** |
| `env_incompatible` | 3 | 解释器或依赖不对（含 `import torch` 失败这类"元数据在、导入炸"的情形） |

两层：`--level eval-now` 只要检查点+资产+协议+入口；`--level full-repro`/`--require-evidence`
再要求 TASK-024 的 20 个种子目录、TASK-026 的 3 条臂、候选包。

### 35.6 本轮实跑证据（窗口 `camra-locomotion:T034`，主机 tiger12.ios.ac.cn）

| 步骤 | 结果 |
| --- | --- |
| `bash -n scripts/repro/*.sh` | 7 个脚本全过 |
| `pytest tests/ -q` | **72 passed** |
| `check_repro_readiness --level full-repro` | **VERDICT: ready**，22 项全 ok |
| `artifact_index build` → `verify` | 10 条目，**PASS** |
| `repro/select_checkpoint.py` 重放 | **iteration 2999，score 0.414823**，6 个候选的 score/rmse_vx/failure_rate 与 `selection_r4.json` 逐项相同 |
| `aggregate_final_test.py` 重放 | 与冻结包 `results/task024_aggregate.txt` **逐字节一致**（仅过滤工具自己追加的 `[agg]` 行） |
| 从 `/tmp/<tmpdir>` 运行预检与选模 | 均 **EXIT 0**，证明路径解析不依赖 cwd |

日志全在 `/pub/data/caohy/logs/camra/TASK-034/`：`t034_run.log`（12 个 `EXIT_` 全 0）、
`readiness_full.json`、`index_verify.log`、`selection_compare.log`、`t024_diff.log`（空 diff）。

### 35.7 本轮踩到的三个坑

1. **`dev_it*_r4` 这个 glob 会吞掉 10 种子目录**。`TASK-023` 下同时有单种子的
   `dev_it<iter>_r4`（6 个）和 10 种子复核的 `dev_it2999_s10x_r4`（10 个），
   原选模脚本按后缀匹配，于是把 `2999_s100` 当迭代号去 `int()`，直接崩。
   现在用 `^dev_it(\d+)<suffix>$` 精确匹配，并把跳过的 28 个目录（6 无后缀 + 6 `_r1` + 6 `_r3` + 10 `_s10x`）
   打印出来——**"跳过了什么"必须可见**，否则又是静默改变口径。
2. **CRLF 会让 bash 在 `case` 处报语法错**。用 Windows 侧重写文件后 `_common.sh` 变成 CRLF，
   `bash -n` 报 `syntax error near unexpected token newline`，指向 `case "$_k" in ''|\#*)`。
   检查方法：`python -c "print(open(p,'rb').read().count(b'\r'))"`，应为 0；
   修法：`sed -i 's/\r$//'`。已确认 `scripts/repro/*.sh`、`config/repro.env` 全为 LF。
3. **`tmux display-message -p '#S:#W'` 报的是会话当前窗口，不是命令所在窗口**。
   第一次记录写成 `camra-locomotion:TASK-023-r3`，与实际运行的 `TASK-034` 不符。
   必须带 `-t "$TMUX_PANE"`。这类"记录看起来对、其实是别处的窗口"正是验收要防的东西。

### 35.8 明确没做的

* 没有把模型、原始日志、冻结包加入 git，也没有上传到任何新服务；索引里只有路径、大小、散列；
* 没有替换资产内容，也没有动其他机器人任务：`assets_go2` 树散列与文件数（4）保持原样；
* 没有改协议、阈值、检查点或训练配置；本轮不训练、不重测（正式复测是 TASK-036）；
* 冻结包 v2 未重新生成——`BUNDLE_SCRIPTS` 增加 `scripts/camra_config.py` 只影响**后续新版本**；
* 干净目录上的完整四步实测留给 TASK-035/037，本任务只证明"路径可解析、工具可复现"。

## 36. 回归测试、接口契约与小批验证（TASK-035）

### 36.1 做了什么

四件事，全部在 tmux 窗口 `camra-locomotion:T035` 里真跑：

| 步骤 | 命令 | 结果 |
| --- | --- | --- |
| 纯 CPU 回归 | `pytest tests/ -q` | **88 passed**（原 72 项未删改 + 新增 16 项） |
| 接口契约 | `verify_runtime_contracts.py --num_envs 4` | **9 项全 PASS** |
| 小批仿真 | `CAMRA_LOG_ROOT=<新目录> repro/finaltest.sh <ckpt> nominal 100 3 32` | 10 场景 / 15 变体 / 45 回合，全产出 |
| 逐步复算 | `recompute_metrics.py --run_dir <小批目录>` | **810 项比对，0 不匹配** |
| 产物审计 | 外部审计脚本（覆盖/方向/簿记/证据计数/终止） | **PASS** |
| 换目录可用性 | 从 `/tmp/<tmpdir>` 调 `check_repro_readiness` / 选模 | **EXIT 0** |
| 输出目录策略 | 同上，`--out-dir` 指向已用目录 | **EXIT 2**（按设计拒绝） |

新增的 16 项测试不是"为了凑数"，每一项都对应一次真实故障或一条必须固定的判分算术：

| 测试 | 固定的缺陷/算术 |
| --- | --- |
| `test_config_*` ×4 | `repro.env` 三级解析顺序（env > file > default）；顺序反了会静默用别的机器的资产根 |
| `test_go2_paths_follow_the_configured_asset_root` | 资产根改名后 `Go2/usd/go2.usd` 仍要正确解析 |
| `test_selection_ignores_per_seed_sweep_directories` | `dev_it2999_s100_r4` 曾被当成迭代号 "2999_s100" 而崩溃 |
| `test_selection_reports_when_no_candidate_meets_the_constraint` | 没有候选满足失败率约束时必须退出码 4，不许静默挑一个 |
| `test_failed_flag_survives_text_forms` | `int("False")` 崩、`bool("0")` 为真；现在 `0/1/False/True` 同义，且手算 0.50 与均值 0.145 对得上 |
| `test_missing_group_is_incomplete_not_a_pass` | 缺一个分组必须 incomplete（退出码 2），不能只算已有的那个 |
| `test_tree_hash_sees_added_removed_and_changed_files` | 树散列必须对增/删/改都敏感，否则换掉一整套文件也发现不了 |
| `test_index_verify_detects_a_swapped_file` / `..._refuses_a_missing_index` | 索引被替换 → 4；索引缺失 → 2 |
| `test_*_classify*` ×3 | 缺资源判 `infra_blocked`(2)、环境坏判 `env_incompatible`(3)、提示项不阻塞 |
| `test_repro_scripts_do_not_assign_bash_reserved_variables` | 见 36.2，静态禁止对 bash 保留变量赋值 |

边界覆盖沿用并补强既有测试：回合数小于/等于/大于并行数、2 变体与 4 变体、非整除配额、
异步终止、NaN/Inf、缺键、空轨迹、合法 N/A、缺场景/缺分组。判分用手算夹具核对（不是"跑通即通过"）。

### 36.2 抓到的真缺陷一：`GROUPS` 是 bash 保留变量

`finaltest.sh` 原本把分组列表存进 `GROUPS`。`GROUPS` 是 bash 的**特殊数组变量**（当前用户的组 ID），
对它的赋值**静默失效**。于是：

```
[finaltest] ckpt=… groups='39400005' seeds='100' episodes_per_variant=3 envs=32
eval_locomotion.py: error: argument --group: invalid choice: '39400005' (choose from 'nominal', 'perturbed')
```

`39400005` 就是这台机器上 `caohy` 的 gid。更糟的是：检查点散列、provenance 都已经写进输出目录，
失败却发生在参数解析，看起来像"仿真器起不来"。

- 修法：变量改名 `GROUP_LIST`；`GROUP_LIST` 不是保留名。
- 防复发：新增静态测试，扫描 `scripts/repro/*.sh` 禁止对
  `GROUPS/UID/EUID/PPID/RANDOM/SECONDS/LINENO/FUNCNAME/BASH_SOURCE/PIPESTATUS/REPLY` 赋值。
- `bash -n` 与 shellcheck 都**不报**这个错，只能靠这类测试。

顺带把最终测试的种子参数改成接受 `300-309` 这种区间写法（文档里的命令可以直接复制）。

### 36.3 抓到的真缺陷二：失败标记的文本解析

`aggregate_final_test.py` 原先写 `sum(int(float(r["failed"]) > 0.5))`：

* `float("True")` 抛 `ValueError`；
* 换成"解析不了就当假"的更宽松写法，会把 `"True"` 读成**未失败**——全线摔倒也能报 0% 失败率。

现在的 `as_flag()` 把 `1/true/yes` 与 `0/false/no/空/NaN` 都写明，其余走数值比较，并逐条写在测试里
（4 回合里 2 个失败 = 0.50，不是 0.00）。

### 36.4 计数口径修正：每变体，而不是每场景

`--episodes_per_scenario` 是 TASK-031 之前的旧口径，现在它会被换算成
`episodes_per_scenario // num_variants`。也就是说给 `combined`（4 变体）写 30，实际每变体只有 7 回合。
`finaltest.sh` 与 `devsweep.sh` 都改为直接传 `--episodes_per_variant`，
README 里也写明第 4 个参数是"每变体"。补充协议 D-2 要求每变体 300，就是 `30 × 10 种子`。

### 36.5 小批到底验证了什么（不是性能结论）

| 场景 | 变体 | 每变体回合 | 首步命令 |
| --- | --- | --- | --- |
| `stand` | 1 | 3 | `(0, 0, 0)` |
| `fwd_low` / `fwd_mid` / `fwd_high` | 1 | 3 | `(0.3/0.6/1.0, 0, 0)` |
| `backward` | 1 | 3 | `(-0.5, 0, 0)` |
| `lateral` | 2 | 3 | `(0, +0.3, 0)` / `(0, -0.3, 0)` |
| `turn_inplace` | 2 | 3 | `(0, 0, +0.8)` / `(0, 0, -0.8)` |
| `combined` | 4 | 3 | `(0.6,0.3,0.5)` / `(0.6,-0.3,-0.5)` / `(-0.6,0.3,-0.5)` / `(-0.6,-0.3,0.5)` |
| `start_stop` / `ramp` | 1 | 3 | 起始为 `(0, 0, 0)`（程序定义如此） |

方向命令是从 `raw/nominal/<scenario>.npz` 的**首步**读回来的，不是从配置推断。
每变体 `accepted=3`、`rejected` = 剩余槽位，`step_guard_hit=False`，说明逐变体配额生效且未提前退出。
**这 45 个回合只用来验证工具，不构成控制器性能结论**，性能结论在 TASK-036。

### 36.6 阈值比较是精确浮点

判分用 `value <= threshold` 的精确比较。`0.1+0.2+0.1+0.2` 在二进制浮点里是 `0.15000000000000002`，
比 0.15 大一个 ulp，因此"数学上刚好等于阈值"的样本会被判**不通过**。这是自协议固定以来一直如此的实现，
本轮**没有**为测试方便加容差；测试夹具因此避开精确边界，并在测试 docstring 里写明原因，
避免以后有人"顺手加个 epsilon"却以为没改判据。

### 36.7 本轮实跑证据

```
/pub/data/caohy/logs/camra/TASK-035/
├── t035_run.log                     # 各步 EXIT：syntax 0 / pytest 0 / contracts 0 /
│                                    #   small_batch 0 / recompute 0 / audit 0 /
│                                    #   readiness_from_tmp 0 / refuses_used_dir 2(期望)
├── pytest.log                       # 88 passed
├── contracts.json                   # 9 项 pass=true（A/B/C/D/E/F 全组）
├── small_batch.log                  # 交付模型、种子 100、每变体 3 回合
├── recompute.log                    # 810 comparisons, 0 mismatches
├── audit.log                        # PASS: coverage, directions, bookkeeping, evidence, endings
├── readiness_from_tmp.log           # 从 /tmp 调仓库内入口，VERDICT: ready
├── readiness_refuses_used_dir.log   # output_dir FAIL，退出码 2
└── batch_<时间戳>/repro/finaltest/nominal/s100/   # summary.json / per_episode.csv / raw/ / recompute_check.json
```

### 36.8 明确没做的

* 没有改判据、阈值、配额或冻结契约；接口契约全部 PASS，因此不需要新建冻结版本；
* 小批不接触最终的保留测试集（用的是开发种子 100，不是 300–309）；
* 没有做任何训练或调参；
* TASK-036 用的评估种子 300–309 在本任务中**一次都没被用过**，仍然干净。

## 37. 确认性复测：9000 回合与逐变体判定（TASK-036）

### 37.1 现场与结论

| 项 | 值 |
| --- | --- |
| 主机 / 窗口 | `tiger12.ios.ac.cn`，`camra-locomotion:T036` |
| 工作目录 | `/home/caohy/repositories/Paper/CaMRA/locomotion`，git `675784b`（工作区干净） |
| 起止 | 2026-11-03T11:23:49+08:00 → 12:19:34+08:00，**55 分 45 秒** |
| 检查点 | `model_2999.pt`，sha256 `1c8f5aad38c2b763…`，与冻结包 `model/model_2999.pt` **字节相同**（`cmp` 通过） |
| 协议 / 补充协议 | `0afc2cea…` / `f50d8b7f…`（与产物索引登记值一致） |
| 评估代码 | `eval_locomotion.py 65d3d3ac…`、`eval_metrics.py 8d63bb5f…`（开跑前逐一散列） |
| 种子 / 配额 | 评估种子 300–309，每种子每变体 30 回合 → 每变体 300；`num_envs 64` |
| 总量 | 15 变体 × 300 × 2 组 = **9000 回合**；输出 `TASK-036/{nominal,perturbed}/s3xx/` |

实际耗时 55 分 45 秒，比补充协议登记的「约 2h57m」短得多：那个估算按 1.18 s/回合线性相加，
没有计入 64 个环境并行——每回合 10 s 仿真时间只有 500 个控制步，GPU 上是 0.37 s/回合。

各步退出码（原始行在 `t036_run.log`）：

```
EXIT_model_matches_frozen=0   EXIT_readiness=0   EXIT_finaltest=0
EXIT_aggregate=1              EXIT_recompute=0   EXIT_report=0
```

`EXIT_aggregate=1` **不是崩溃**：`aggregate_eval_runs.py` 用 0=完整且通过、1=完整但未通过、2=不完整，
这里 1 正是"证据齐了、判定没过"。

| 判定项 | 结果 |
| --- | --- |
| 运行完整度 | `runs_present=20 / runs_expected=20`，`complete=True`，`missing_runs=[]` |
| 证据不足 | **0** 个「组×变体×指标」低于 90% 有效下限，全部 300/300 有效 |
| 逐变体 | **29 / 30 达标** |
| 场景 | nominal 9/10、perturbed 10/10 |
| 失败回合 | nominal 0/4500、perturbed 17/4500（0.0038，限值 0.10） |
| `overall_pass` | **False** |

唯一不达标项是 **nominal `lateral#v1`（`vy = −0.3 m/s`）**：

| 指标 | 值 | 阈值 | 余量 |
| --- | --- | --- | --- |
| `rmse_vy` | 0.1035 | 0.1000 | **超 3.5 %** |
| `p95_abs_err_vy` | 0.1361 | 0.1800 | 通过 |
| `failure_rate` | 0.000 | 0.050 | 通过 |

同方向在 perturbed 是 0.1134 ≤ 0.1200（父协议放宽 20 %）通过；反方向 `lateral#v0`（`vy=+0.3`）
在 nominal/perturbed 分别是 0.0887 / 0.1091。**同一模型左右两个横向方向差 17 %**，
且不掉、不摔、不超时（`failure_rate=0`），所以这不是稳定性问题，是单方向跟踪精度差了 3.5 %。

### 37.2 旧最终测试到底测了哪些方向

TASK-024 的 `summary.json` 是**场景级**的：一个场景一行，没有变体字段。改成逐变体后，第一件要回答的事是
"旧的那 10/10 到底覆盖了协议的哪些变体"。答案不在 summary 里，要从原始轨迹读。

旧 `raw/<group>/<scenario>.npz` 有 `steps`（逐步矩阵）和 `lengths`（每回合步数），`fields` 里含
`cmd_vx/cmd_vy/cmd_wz`，但**没有** `variant_ids`。按 `lengths` 逐回合切片、取每回合第 0 步的命令三元组，
就能把"实际下发过的命令组合"数出来。对 10 个旧种子、2 个分组、3 个多变体场景逐一统计（300 回合/格）：

| 场景 | 协议变体数 | 旧 TASK-024 实际下发 | 旧覆盖 |
| --- | --- | --- | --- |
| `lateral` | 2 | `(0, +0.3, 0)` × 300 | **1 / 2** |
| `turn_inplace` | 2 | `(0, 0, +0.8)` × 300 | **1 / 2** |
| `combined` | 4 | `(+0.6, +0.3, +0.5)` × 300 | **1 / 4** |
| 其余 7 个场景 | 各 1 | 与协议一致 | 1 / 1 |

新 TASK-036 同法统计（s300）：`lateral` 两个方向各 30、`turn_inplace` 两个方向各 30、
`combined` 四个对角组合各 30——15 个变体全到。

即：旧最终测试在 15 个协议变体里实际覆盖 **10 个**（7 个单变体场景 + 3 个多变体场景各 1 个方向）。
这不是新发现的缺陷，而是 TASK-030 裁定的口径问题的**可量化边界**：旧结果按旧协议仍记 PASS，
但它 PASS 的是 10 个变体，不是 15 个。

### 37.3 新旧数字哪些对得上、哪些对不上

同一模型、同一场景、单变体条件下重跑，数字几乎逐位重合（旧 → 新合并）：

| 场景 | 指标 | 旧 | 新 | 差 |
| --- | --- | --- | --- | --- |
| `fwd_low` | `rmse_vx` | 0.1331 | 0.1330 | −0.0001 |
| `fwd_mid` | `rmse_vx` | 0.0665 | 0.0665 | 0 |
| `fwd_high` | `rmse_vx` | 0.0533 | 0.0537 | +0.0004 |
| `backward` | `rmse_vx` | 0.0573 | 0.0573 | 0 |
| `ramp` | `rmse_vx` | 0.1096 | 0.1099 | +0.0003 |
| `stand` | `stand_drift_m` | 0.0036 | 0.0036 | 0 |
| `start_stop` | `rise_time_s` | 0.2201 | 0.2203 | +0.0002 |

七个单变体场景全部在 1e-4 量级内复现。这同时证明两件事：模型没变、评估链路没走样。

三个多变量场景则对不上，原因只有一个——**测量对象变了**：

| 场景 | 旧（只 1 个方向） | 新合并（全方向） | 新逐变体 |
| --- | --- | --- | --- |
| `lateral` | 0.0887（`+0.3`） | 0.0961 | 0.0887 / **0.1035** |
| `turn_inplace` | 0.0707（`+0.8`） | 0.0580 | 0.0707 / 0.0453 |
| `combined` | 0.0609（`(+,+,+)`） | 0.0722 | 0.0609 / 0.0619 / 0.0833 / 0.0828 |

`lateral` 合并值比旧值**高**（加进更难的反方向），`turn_inplace` 合并值比旧值**低**
（加进更易的反方向）——合并值往哪边走取决于新加的方向是难是易，所以"D-4 逐变体"不是形式要求：
按合并值判，`lateral` 0.0961 ≤ 0.10 会通过，翻车的那个变体就被平均掉了。

### 37.4 复算与证据

- `recompute_metrics.py` 对 20 个种子目录逐一从 `raw/*.npz` 逐步轨迹重算：**20 PASS / 0 FAIL**，
  共 **162 000** 项指标比对、0 不匹配（每目录 8100 项，含 10 场景 × 各自指标 × 回合）。
- 报告 `report.md` 八节：结论、口径、逐变体表（30 行）、场景合并值（20 行）、证据不足明细（空）、
  复算汇总、与旧结果对照、仍未覆盖。**单模型**在 §1 与 §2 各写一次。
- 证据树 934 MB：`preflight/`（5 个散列与预检输出）、`finaltest.log`、`aggregate.{json,log}`、
  `recompute.log`、`report.md`、`{nominal,perturbed}/s3xx/`。

### 37.5 报告工具自身的两个缺陷（本轮发现并修）

产物是证据，所以工具的渲染错误也要修，但**只改渲染、不改判定**：

1. 逐变体表的「状态」列全部渲染成 `?`。原因是聚合 JSON 中变体层没有 `status` 字段，
   状态记录在下挂的每个指标 check 里，`v.get('status', '?')` 永远取默认值。
   改为按 check 归并：未完成 → 证据不足 → 不达标 → 达标，依次优先。
2. 第 7 节只比 `rmse_vx`，于是 `lateral`/`turn_inplace`/`stand`/`start_stop` 全是"—"，
   看上去像是"旧结果没有这些数字"。旧 summary 记的是该场景被判的指标，不是固定的 `rmse_vx` 列；
   改为按指标名 join 后，`lateral` 的 `rmse_vy` 才显示出来（0.0887 → 0.0961）。

两处修改后重新生成 `report.md`，`overall_pass` 前后都是 `False`——修的是可读性，不是结论。

### 37.6 结论与建议

**结论：未通过（`overall_pass=False`）。** 依补充协议 D-4，nominal `lateral#v1` 不达标即该场景不通过，
nominal 组要求全部场景通过，因此整体不通过。不得据此宣称已具备 calibration 交付条件。

用户裁决建议（三者互斥，任选其一；本任务不自行动手）：

1. **接受为已声明边界**：把"nominal `vy=−0.3` 的 `rmse_vy` 0.1035，超阈值 3.5 %，无摔倒"写进交付说明，
   下游校准按这个残差水平设计裕度。代价最小，但 `overall_pass` 仍是 False。
2. **先查左右不对称的成因**（CPU 侧、不动模型）：核对奖励项与命令采样分布是否左右对称、
   观测里 `base_ang_vel` 与命令的耦合、步态相位偏好。若是对称性缺陷，修配置比补训更省。
3. **定向修复后重测**：任一新模型版本都要**重新登记测试种子**（300–309 已用掉，不得复用，
   否则等于拿已看过的数据挑优），并按 TASK-037 要求出独立版本、重跑导出一致性与闭环重载。

明确不做：不重跑挑优、不换种子、不调阈值、不把这次结果当"再调一版参数"的依据。

### 37.7 明确没做的

- 没有补训、没有改奖励/课程/阈值/配额，`eval_protocol.json` 与补充协议逐字未动；
- 没有用 300–309 之外的种子，也没有在看完结果后追加任何一次运行；
- 没有把 9000 回合原始日志放进 git（934 MB，按 `docs/reproduction.md` 的边界留在仓库外）；
- 冻结包 v2 未重新生成，检查点字节未变。

## 38. 收尾：交付说明、干净克隆复核与未关闭项（TASK-037）

### 38.1 文档改了哪些

本轮不产生新模型、不产生新冻结包，因此所有说明都锚在**同一个模型散列**上：
`model_2999.pt` sha256 `1c8f5aad38c2b763…`（检查点与冻结包内副本字节相同），
协议 `0afc2cea…`、补充协议 `f50d8b7f…`。

| 文件 | 改动 | 为什么 |
| --- | --- | --- |
| `CaMRA/locomotion/README.md` | 交付状态表增加「确认性复测（TASK-036）」行；旧最终测试行标注为**旧口径**并写明它只测了 10/15 个变体；表后新增判定段 | 接手的人第一眼要看到"哪个数字是什么口径"以及"当前结论未通过" |
| `CaMRA/README.md` | locomotion 行改为「训练、评估、导出与冻结已完成，确认性复测未通过验收」；calibration 行补「进入校准前需先裁决处置方式」 | 模块级状态不能停在"已完成" |
| `docs/implementation_handoff.md` | 新增第 37、38 章 | 复测现场、覆盖边界、新旧对照、结论与建议 |
| `docs/reproduction.md` | 新增第 6 节「干净克隆实测」 | TASK-034 遗留的四步走查，本轮补做并留证 |
| `scripts/aggregate_eval_runs.py` | 用法示例里的 `/pub/data/caohy/logs/camra/TASK-036` 改为 `"$CAMRA_LOG_ROOT/TASK-036"` | 示例不该教人抄一台机器的绝对路径 |

旧记录一律保留、只追加：TASK-024 的 PASS、第 31 章的覆盖边界、第 36 章的回归结论都没有被改写，
新增的只是"哪次任务澄清了什么"。

### 38.2 干净克隆复核（提交 `9bbec29`）

在 `tmux` 窗口 `camra-locomotion:T037CL` 内运行，整套 34 秒完成，日志 `TASK-037/cl_run.log`：

| 步骤 | 命令 | 实测 |
| --- | --- | --- |
| 克隆 | `git clone` → `rev-parse --short HEAD` | `9bbec29`，`git status --porcelain` 0 行 |
| 预检（不导出任何 `CAMRA_*`） | `env -u CAMRA_* python scripts/check_repro_readiness.py --level eval-now` | `VERDICT: ready`，退出码 0，19 项检查全 ok |
| 覆盖优先级 | `CAMRA_ASSET_ROOT=/nonexistent/asset … readiness` | `VERDICT: infra_blocked`，**退出码 2**，blocking=`asset_root, go2_usd` |
| 索引校验 | `python scripts/artifact_index.py verify` | `PASS: all 10 entries match`，退出码 0 |
| 索引缺失 | `… --index /nonexistent.json` | 退出码 2，`FATAL: index not found` |
| 入口点 | 12 个 `.py` 跑 `--help`、7 个 `.sh` 跑 `bash -n` | **17 个返回 0**；`export_policy_candidate.py`、`rollout_exported_policy.py` 返回 2 |
| 冻结模型 | `sha256sum …/v2/model/model_2999.pt` | `1c8f5aad38c2b763…`，与索引登记值一致 |

两个 `--help` 返回 2 的原因已在 `docs/reproduction.md` §6.1 写明：IsaacLab 的
`add_app_launcher_args()` 会把 `-h/--help` 从 `parser._option_string_actions` 摘掉，
只在检测到既有 help action 时补回，本环境没走到补回分支，于是 `--help` 落到"缺必填参数"。
这是上游行为，不是入口坏了——两者的可执行性由 TASK-025/026 实跑证明。

验证前后都核对了冻结包：模型散列未变，`artifact_index.py verify` 的 10 条全部匹配
（其中 `frozen_bundle_v2` 树散列 `8923c6fd4f05a921…`、31 文件、29,127,046 B）。

提交 `b62aca7`（即本轮最终文档状态）在一个新克隆里又复跑了关键三项：预检 `ready`、产物索引 10 条 PASS、交接笔记 38 章在册（`chapters=38`），日志 `TASK-037/cl_recheck_b62aca7.log`。这样文档写的和干净目录里跑出来的是同一份提交。

### 38.3 v2 冻结包为什么没有新版本

本轮的改动全部落在**文档、说明和评估入口**上：没有改执行时序、预处理、动作映射、增益或命令域，
`eval_protocol.json` 与补充协议逐字未动，检查点字节未变。按 TASK-037 的规则，这种情况保持 v2
逐字节不变、用独立版本的验证证据（本章 + 第 37 章）关联其散列即可；**没有**为了"看起来有进展"
而重导出、重打包或新建版本号。

### 38.4 计划状态

| 任务 | 状态 | 理由 |
| --- | --- | --- |
| TASK-029 … TASK-035 | 完成（7 项） | 见第 31–36 章 |
| **TASK-036** | **未通过（false）** | 实施完整、证据齐全（20/20 种子目录、9 000 回合、162 000 项复算 0 不匹配、0 证据不足），但按 D-4 逐变体判据 29/30 达标，`overall_pass=False`。按任务要求"保留结果、保持 false、提出建议、等待授权" |
| **TASK-037** | **未通过（false）** | 文档与复核工作已完成；但其验收条件含"新增 9 项各有实证**并满足验收**"，TASK-036 未满足，故不宣称本轮验收通过 |

计划版本 `3.6.0 → 3.7.0`，任务完成数仍为 **35/37**——两个补充任务有完整工作记录但验收未通过，
不用 `status=true` 掩盖结论。

### 38.5 未关闭项与进入 calibration 的前置条件

**前置条件（未满足）**：TASK-036 的逐变体验收未通过，因此**当前不具备进入 calibration 的条件**。
不得用"旧口径 10/10 场景通过"或"29/30 变体达标"替代这一结论。

需要用户裁决的处置方式（三条互斥，详见 §37.6）：接受为已声明边界 / 先查左右不对称 / 定向修复后重测。
第三种方式必须**重新登记测试种子**：300–309 已在本次复测中用掉，复用即为拿看过的数据挑优。

本轮仍未覆盖（随交付一起声明）：0–40 ms 指令延迟、真机部署与 `joint_ids_map` 重映射、
复杂地形、高于协议的命令域、真机状态估计噪声与执行器时延、训练种子间方差（只有 1 个训练模型）。

### 38.6 九项补充任务对账

| 任务 | 做了什么 | 结论 |
| --- | --- | --- |
| TASK-029 | 只读复核既有验收证据 | 发现模型数、变体覆盖、指标完整性三处问题 |
| TASK-030 | 定四项裁决（单模型、每变体 300、90 % 有效、逐变体判定） | 补充协议 v1.0.0 生效，sha256 登记 |
| TASK-031 | 逐变体配额调度 | 修复多变体场景只测一个方向的缺陷 |
| TASK-032 | 逐指标有效数门控 | "1 有效 + 29 缺失"不再被当成 30 个有效 |
| TASK-033 | 停车概率渐增的时间单位 | 注释纠错，行为不变 |
| TASK-034 | 仓库内复现入口与产物索引 | 18 个入口可用，10 条产物散列可校验 |
| TASK-035 | 回归测试、接口契约、小批验证 | 88 项测试、9 项 50/200/10 Hz 契约、810 项复算 |
| TASK-036 | 9 000 回合确认性复测 | **未通过**：nominal `lateral#v1` 超阈值 3.5 % |
| TASK-037 | 交付说明、干净克隆复核、收尾 | 文档完成，验收随 TASK-036 一并保持未通过 |

## 39. 低速工况的稳态缺口（TASK-036 裁决②诊断）

TASK-036 判不通过后，用户选择「先查左右不对称的成因」。诊断只读 `TASK-036` 的既有 npz，
不跑仿真、不动模型、不改协议。结论和最初的假设不一样：**左右不对称是真的，但不是主因；真正的弱区是低速单轴**。

### 39.1 先把 0.1035 拆开

`rmse² = bias² + std²` 是恒等式，拆完就知道是"系统性少走"还是"抖得厉害"。窗口与官方复算一致
（`episode_step × step_dt > 1.0` 才计入），所以数字与报告里的 rmse 对得上。

| 组 | 变体 | rmse | bias（有符号） | std | bias 占 rmse² | 命令 | 稳态实测 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| nominal | `lateral#v0` | 0.0887 | −0.0870 | 0.0174 | **96.2 %** | +0.3 | +0.213 |
| nominal | `lateral#v1` | 0.1035 | +0.0997 | 0.0278 | **92.8 %** | −0.3 | −0.200 |
| perturbed | `lateral#v0` | 0.1119 | −0.0977 | 0.0466 | 76.2 % | +0.3 | +0.202 |
| perturbed | `lateral#v1` | 0.1142 | +0.1029 | 0.0460 | 81.1 % | −0.3 | −0.197 |

两条直接结论：

1. **不是抖动**，是稳态缺口——偏置吃掉误差平方的 93–96 %；
2. **两侧都在缺**：往右只跑到命令的 71 %，往左只跑到 67 %。这不是"往左不行往右行"。

### 39.2 不是收敛慢，是到顶了

把 500 步按 1 秒分箱（`diag_lateral_time.py`）：

| 窗口 | v0 实测 vy | v1 实测 vy |
| --- | --- | --- |
| 0–1 s（还在 settle） | +0.183 | −0.165 |
| 1–2 s | +0.216 | −0.201 |
| 2–3 s | +0.213 | −0.197 |
| 5–6 s | +0.213 | −0.203 |
| 9–10 s | +0.212 | −0.204 |

第 2 秒就到了稳态，第 10 秒一模一样。**10 秒的回合长度不是瓶颈**，加长评测没有用。

同一份输出也排除了"变体绑死在某批环境槽位"：v0 用 30 个槽位（0–58）、v1 用 30 个槽位（1–59），
两者交织，不是前后半分。十个种子的 rmse 也几乎不动（v0: 0.0886–0.0888，v1: 0.1031–0.1040），
说明误差由确定性因素主导，不是随机扰动抽签。

### 39.3 对照其他场景，弱区是"低速"而不是"侧向"

同一次复测的其它场景用同样方法拆一遍（nominal）：

| 场景 | 命令 | 稳态实测 | 相对缺口 | rmse | bias 占比 | yaw 漂移 (°/s) |
| --- | --- | --- | --- | --- | --- | --- |
| `fwd_high` | vx 1.0 | 0.957 | 4 % | 0.0538 | 64.7 % | +0.10 |
| `fwd_mid` | vx 0.6 | 0.536 | 11 % | 0.0665 | 92.7 % | +0.13 |
| `backward` | vx −0.5 | −0.445 | 11 % | 0.0573 | 93.9 % | −0.15 |
| `lateral#v0` | vy +0.3 | 0.213 | 29 % | 0.0887 | 96.2 % | −0.05 |
| `lateral#v1` | vy −0.3 | −0.200 | 33 % | 0.1035 | 92.8 % | **+0.46** |
| `fwd_low` | vx 0.3 | 0.169 | **44 %** | 0.1330 | 97.1 % | **+1.27** |
| `combined`（+vy） | vy +0.3，vx 0.6 | +0.271 | 10 % | 0.0387 | 58.0 % | −0.30 |
| `combined`（−vy） | vy −0.3，vx 0.6 | −0.279 | 7 % | 0.0321 | 41.2 % | −0.40 |
| `ramp` | 斜坡到 0.6 | — | — | 0.1100 | **6.3 %** | +0.01 |

三件事：

1. **0.5 m/s 以上缺口 ≤ 11 %，0.3 m/s 时 29–44 %。** 侧向 0.3 只是落在低速区，不是"侧向特别差"。
   `fwd_low`（vx=0.3）缺 44 %，比 `lateral#v1` 还差——如果按同样的阈值口径，`fwd_low` 的 rmse 0.1330
   对阈值 0.15 只剩 11 % 余量，是下一个会撞线的场景。
2. **`combined` 是判据的关键反例**：同一个 vy=0.3，伴随 vx=0.6 时只缺 10 %、rmse 0.039。
   所以"侧向能力只有 0.2"的假设不成立——策略在有前进分量时能把侧向跟到 0.28。
   决定误差的是**总速度模长**（combined 为 0.84，纯侧向为 0.3），不是哪个轴。
3. **`ramp` 反向验证了方法**：bias 只占 6.3 %、std 0.106，是动态跟踪滞后（斜坡命令跟不上），
   与稳态缺口是两类误差。同一个脚本能把两者分开，说明 39.1 的结论不是脚本口径造成的。

### 39.4 剩下的那个"左右不对称"有多大

在低速弱区的背景下，左右差异是**次要但真实**的：

| 指标 | v0（vy=+0.3） | v1（vy=−0.3） | 差异 |
| --- | --- | --- | --- |
| bias | −0.0870 | +0.0997 | v1 多缺 15 % |
| std | 0.0174 | 0.0278 | v1 波动大 60 % |
| yaw 漂移 | −0.047 °/s | **+0.460 °/s** | 10 秒差约 4.6° |
| 左右腿动作幅值比 | 0.753 | 0.762 | **没有镜像反转** |

最后一行值得单独说：`|a|左 / |a|右` 在两个方向上都约 0.75，**没有随命令方向翻转**。
如果模型左右对称，往左走和往右走时左右腿的用力关系本该对调。现在不管往哪走都是右侧动作幅值更大，
说明策略里存在静态的左右偏置，而不只是"往左走得更差"。

它的量级不足以单独解释 0.1035（bias 只差 0.013），但它决定了**擦线还是翻车**：
v0 的余量只有 0.1000 − 0.0887 = 0.0113，v1 超了 0.0035。**只要低速能力不改善，两侧都贴着阈值走**。

### 39.5 已排除的方向

| 怀疑 | 检查 | 结论 |
| --- | --- | --- |
| 训练命令分布左右不对称 | `commands.base_velocity.limit_ranges = (-1,1)/(-0.4,0.4)/(-1,1)`，均匀重采样 | 对称，排除 |
| 奖励项有方向性 | 16 个基线奖励项 + 新增 2 项，无按轴向的权重 | 排除 |
| 变体与环境槽位混淆 | v0 槽位 0–58、v1 槽位 1–59，交织 | 排除 |
| 评测时长不够 | 分箱显示 2 s 达稳态 | 排除 |
| 随机扰动抽签 | 十种子 rmse 变动 <1 % | 排除 |
| 侧向能力上限 | combined 里同幅值只缺 10 % | 排除 |

### 39.6 处置建议（三选一，都需要用户决定）

1. **治本：重训低速区间。** 侧向/低速的稳态缺口是策略稳定地选了一个"少走"的平衡点
   （std 只有 0.017–0.028，不是控制不住），典型成因是奖励对 0.1 量级误差不够敏感
   （`track_lin_vel_step` 的 `std=0.15`，误差 0.1 时已进入 exp 的平坦段），
   叠加低速步态本身效率低。可行改动：收紧 `std`、给低速单轴加采样权重或课程、检查低速打滑。
   任何一条都会改变行为 → 新模型版本 + 重新登记测试种子。
2. **改口径：承认阈值与能力不匹配。** 现状是 `lateral` 阈值 0.10 对命令 0.3（33 %），
   而模型低速能力就在 0.2 附近，`fwd_low` 0.133 对 0.15 也只剩 11 % 余量。
   要么把阈值按"命令幅值的比例"重新定义，要么把 `lateral` 命令从 0.3 降到 0.2。
   **这属于放宽验收标准**，必须在新版协议里写明是"匹配能力"还是"为了通过"，不能悄悄改。
3. **接受边界并传给下游。** 在交付说明里写清"低速（≈0.3 m/s）单轴指令的稳态只达到命令的 56–71 %，
   导航/校准在低速区必须留更大裕度或避开纯低速指令"。代价最小，但下游成本函数会吃到这个偏差。

第 3 条对 calibration 尤其重要：**这不是噪声，是系统性偏置**，用平均值校准可以直接吸收一部分，
用方差校准则完全吸收不了。

### 39.7 产物

| 文件 | 内容 |
| --- | --- |
| `/pub/data/caohy/tools/diag_lateral.py` | bias/std 分解 + yaw 漂移 + 左右动作幅值，`--axis` 可选 vx / vy / wz |
| `/pub/data/caohy/tools/diag_lateral_time.py` | 1 秒分箱、稳态分布分位数、环境槽位归属 |
| `TASK-036/diag/lateral_split.json` | 四组 lateral 的分解 |
| `TASK-036/diag/lateral_time.json` | 时间分箱与槽位 |
| `TASK-036/diag/{combined,fwd_low,fwd_mid,fwd_high,ramp,backward}_*.json` | 对照场景 |

两个脚本都只读既有 npz，不启动仿真器，可随时重跑。

## 40. 速度扫描：0.2 m/s 的起步死区（TASK-036 裁决② 续）

第 39 章把问题定位到「低速单轴」。这一章回答下一步：低速到底怎么个差法——是到顶了、按比例缩水，还是别的形状。
答案推翻了我在 §39.6 的假设：**不是精度问题，是起步问题**。

### 40.1 方法

临时协议 `/pub/data/caohy/tmp/protocol_sweep.json`（**不进仓库**，不构成任何协议修订）：
从 `eval_protocol.json` 派生，把 10 个场景换成 16 个单变体常量，`vy = −0.40 … +0.40` 步长 0.05，
`vx = wz = 0`，其余（warmup、阈值、指标定义）逐字沿用。开发种子 100、每点 30 回合、`num_envs 64`。
**评估种子 300–309 一个都没用**（已消耗，不得复用）。

先做冒烟对账：`vy=+0.30` 在种子 100 上得 `rmse_vy = 0.0888`，与 TASK-036 用种子 300–309 得到的
0.0886–0.0888 一致——同模型、同命令、不同种子，说明扫描链路与正式评估同一口径，曲线可直读。

### 40.2 曲线

| 命令 vy | 稳态实测 | 缺口 | 相对缺口 | rmse | std |
| --- | --- | --- | --- | --- | --- |
| −0.05 … −0.15 | ≈ 0.000 | 全额 | 100 % | 0.050–0.150 | 0.0005–0.0025 |
| −0.20 | −0.0007 | 0.199 | **99.7 %** | 0.2010 | 0.0255 |
| −0.25 | −0.1341 | 0.116 | 46.4 % | 0.1183 | 0.0239 |
| −0.30 | −0.2007 | 0.099 | 33.1 % | 0.1031 | 0.0279 |
| −0.35 | −0.2569 | 0.093 | 26.6 % | 0.0987 | 0.0327 |
| −0.40 | −0.3015 | 0.099 | 24.6 % | 0.1045 | 0.0351 |
| +0.05 … +0.15 | ≈ 0.000 | 全额 | 100 % | 0.050–0.150 | 0.0012–0.0122 |
| +0.20 | +0.0059 | 0.194 | **97.1 %** | 0.1959 | 0.0266 |
| +0.25 | +0.1413 | 0.109 | 43.5 % | 0.1167 | 0.0424 |
| +0.30 | +0.2129 | 0.087 | 29.0 % | 0.0888 | 0.0174 |
| +0.35 | +0.2606 | 0.089 | 25.6 % | 0.0906 | 0.0146 |
| +0.40 | +0.3155 | 0.085 | 21.1 % | 0.0855 | 0.0132 |

### 40.3 三件事

**一、|命令| ≤ 0.20 时机器人完全不动。** 不是"走得慢"，是稳态速度 0.000——两侧都是。
0.05 到 0.20 六七个点的实测都在 ±0.006 以内，而 std 只有 0.001–0.027。
这是**死区**，不是弱跟踪。

**二、越过 0.20 后，缺口变成近似固定的 ~0.09。** 0.25→0.116、0.30→0.099、0.35→0.093、0.40→0.099，
不再随命令增大而明显变化。所以命令越大，**相对**缺口越小（0.40 时 21–25 %），但绝对误差不会消失。

**三、`lateral` 的命令 0.30 恰好卡在死区出口，是最难看的那个点。** 相对缺口 29–33 %，
`rmse` 0.089/0.103 正好骑在 0.10 的阈值上——往前一点就过、往后一点更差。
这也解释了 `combined`：那边 vx=0.6 已经在走，侧向只是"边走边挪"，不需要起步，
所以同一个 vy=0.3 只缺 0.03（见 §39.3）。

### 40.4 这推翻了什么

原假设是"奖励对 0.1 量级误差不敏感（`std=0.15`）导致跟踪精度差"，因此打算收紧 `std` 重训。
**曲线不支持这个假设**：精度问题会在全幅值范围内按比例表现，而这里 ≤0.20 是**完全不执行**、
>0.20 是固定缺口。收紧 `std` 改善的是"走得准不准"，治不了"压根不起步"。

也推翻了两个候选处置：

| 原候选 | 曲线给出的判断 |
| --- | --- |
| 把 `lateral` 命令从 0.30 降到 0.20–0.25 | **会变得更糟**：0.20 时 rmse 0.196、0.25 时 0.118，都比 0.30 的 0.089/0.103 差 |
| 把命令提到 0.40 换取更小相对误差 | 正方向 0.0855 略好，但负方向 0.1045 与 0.1031 持平——**不能解决问题** |

### 40.5 更像的成因

死区 + 固定缺口的形状，更像策略学到了"命令小就站着"这条规则，而不是步态能力不足。
训练侧有两处会强化这条规则，值得先查（都不需要重训就能读配置和日志）：

1. `commands.base_velocity.rel_standing_envs`：一部分环境整回合站着，且不是"先走再停"；
2. `commands.base_velocity.stop_event_prob = 0.25`（r4 加入）：25 % 的重采样直接变成停止保持。
   `eval_protocol.json` 的命令域含零，所以这不算改变任务定义，但它让"停"成为高频样本。

若确认是这条规则造成的，修法是**让低速非零命令也进入训练分布**（提高 |v| ∈ [0.1, 0.3] 的采样比例、
或对"命令非零却长时间静止"加惩罚），而不是收紧奖励精度。这仍然是重训，但改的是采样而不是精度项。

### 40.6 保留的不确定性

- 本曲线是**单种子（100）、每点 30 回合**，趋势极清晰（多个点 100 % 缺口），但死区的精确边界
  （0.20 与 0.25 之间）没有细分；若要用于协议决策，需要在 0.20–0.25 之间补点并多种子。
- 表里的 yaw 漂移列是单种子统计，波动大（−2.9 至 +4.4 °/s），**不作为结论**；
  §39.4 那份（−0.047 / +0.460 °/s）才是 10 种子 300 回合的稳健值。
- 死区是否随扰动变化（perturbed 组）未测。

### 40.7 产物

| 文件 | 内容 |
| --- | --- |
| `/pub/data/caohy/tmp/protocol_sweep.json` | 临时扫描协议（16 场景，仓库外） |
| `/pub/data/caohy/scripts/t040_sweep.sh` | 驱动：冒烟对账 → 全量 → 曲线 |
| `/pub/data/caohy/tools/gen_sweep_protocol.py` | 从正式协议派生扫描协议 |
| `/pub/data/caohy/tools/sweep_report.py` | 读 raw npz 出"命令 → 实测"表 |
| `TASK-036/sweep/sweep_run.log` | 现场日志（主机、窗口 `camra-locomotion:T040SWEEP`、git 435b802、各步 EXIT） |
| `TASK-036/sweep/sweep_curve.json` | 16 点结构化结果 |

## 41. 命令采样 A/B：死区被撬开一半（TASK-036 裁决② 续）

第 40 章发现 0.2 m/s 的起步死区，怀疑是"命令小就站住"学出来的。这一章用一次 A/B 验证，
结论比预期复杂：**假设成立，但主角不是改采样，是"训练还没收敛"**。

### 41.1 设计

两组都从交付检查点 `model_2999.pt` 继续训 250 迭代（2999 → 3248），起点相同，只差命令采样：

| 组 | 配置 | 差异 |
| --- | --- | --- |
| A 基线 | `rel_standing_envs=0.1`、`stop_event_prob=0.25` | 交付配置原样 |
| B 无站立 | `rel_standing_envs=0`、`stop_event_prob=0` | 移除两个"零命令"来源 |

改动通过环境变量 `CAMRA_EXP_CMD_SAMPLING=no-standing` 注入（`camra_env_cfg.py` 里的诊断钩子，
**默认不设即完全不变**）。开跑前先用 1 迭代冒烟，从 dump 出的 `params/env.yaml` 确认
`rel_standing_envs: 0.0`、`stop_event_prob: 0.0` 真的进了配置，且覆盖校验没有误报。

评估用第 40 章那份临时协议里的 4 个点（±0.15、±0.30），开发种子 100、每点 30 回合。
**评估种子 300–309 未使用**。

### 41.2 结果

| 命令 | 原模型 (2999) | A 基线 (3248) | B 无站立 (3248) |
| --- | --- | --- | --- |
| +0.15 实测 / rmse | +0.0008 / 0.1497 | +0.0590 / 0.1025 | +0.0649 / **0.0889** |
| −0.15 实测 / rmse | +0.0001 / 0.1501 | −0.0027 / 0.1497 | −0.0797 / **0.0810** |
| +0.30 实测 / rmse | +0.2129 / 0.0888 | +0.2313 / 0.0722 | +0.2384 / **0.0633** |
| −0.30 实测 / rmse | −0.2007 / 0.1031 | −0.2467 / 0.0602 | −0.2472 / **0.0575** |

**一、光继续训练，不达标的项就达标了。**
A 组没改任何配置，只是多训 250 迭代：−0.30 的 rmse 从 **0.1031 降到 0.0602**，
+0.30 从 0.0888 降到 0.0722。也就是说 r4 的 3000 迭代**不是收敛点**，模型还在快速改进期。
TASK-036 判不通过的那一项，靠"训久一点"就能拿到约 40 % 的余量。

**二、去掉站立采样再叠一层，主要作用在真·低速。**
B 组相对 A 组：−0.15 从 −0.0027（不动）到 **−0.0797**（走了 53 %），rmse 0.1497 → 0.0810；
+0.15 rmse 0.1025 → 0.0889；±0.30 再降 12 % 与 4 %。
第 40 章的死区在负方向被撬开了一半，正方向也松动（0.0008 → 0.0649）。

**三、但死区没有消失。** +0.15 在两组都只到命令的 43 %，−0.15 在 A 组仍然几乎不动。
250 迭代只够把"完全不执行"改成"执行一半"，不足以把死区抹平。

### 41.3 这改变了什么

原计划是"按新成因重训低速起步行为"，隐含假设是要动采样、动奖励、要重新设计。
结果说明**最省的动作可能就够了**：继续训练。理由有两条，都能从表里读出来：

- 训练没收敛：250 迭代就有 40 % 的 rmse 改善，说明 3000 迭代停早了；
- 采样调整是"锦上添花"，不是"必要条件"：A 组不动配置已经让 −0.30 从 0.1031 到 0.0602。

对 R0 交付的影响：**TASK-036 的失败很可能不是模型能力问题，是训练预算问题**。
这比"要重新设计奖励"轻得多，也解释了为什么开发集选模（TASK-023）当时选中 2999——
选模只看开发种子 100–109 的场景级指标，挑不出"训练还没收敛"这件事。

### 41.4 必须保留的不确定性

- **单种子（100）、每点 30 回合**。趋势清楚（−0.15 从 0.0027 到 0.0797 是质变），
  但绝对数字会有种子间波动；正式判定仍需 10 种子 300 回合。
- **250 迭代的中间产物，没走选模、没做导出与闭环重载**。表里的数字只能用来判断方向，
  不能当作"新模型已达标"。
- yaw 漂移列在 B 组出现 +2.14 °/s 这类值，属单种子噪声，**不作为结论**。
- 没有测试更长训练是否继续改善（500、1000 迭代），也没有测试"只继续训、不改采样"在
  +0.15 上是否够用。

### 41.5 两个工程坑（都真踩了）

1. **`--max_iterations` 在 rsl_rl 3.0.1 是"再训多少次"，不是"训到第几次"**。
   第一次跑传 3000 想"训到 3000"，实际是"从 2999 再训 3000"，日志显示 `iteration 2999/5999`、
   ETA 三小时，发现后立刻停掉。正确用法是 `--max_iterations 250`。
2. **训练必须在 `$CAMRA_RUN_ROOT` 下跑**。rsl_rl 的日志根是相对路径 `logs`，
   在仓库目录直接跑会去 `CaMRA/locomotion/logs/...` 找 checkpoint 并报 `FileNotFoundError`。
   `scripts/repro/train.sh` 靠 `logs -> $CAMRA_RSL_LOG_ROOT` 软链 + `cd` 解决，手写驱动要照抄。

### 41.6 产物

| 文件 | 内容 |
| --- | --- |
| `camra_env_cfg.py` | 诊断钩子 `CAMRA_EXP_CMD_SAMPLING`（默认关闭）+ 覆盖校验的实验豁免 |
| `/pub/data/caohy/scripts/t041_ab.sh` | 驱动：冒烟验证钩子 → A/B 各 250 迭代 → 四点评估 → 对照 |
| `TASK-036/ab/ab_run.log` | 现场日志（窗口 `camra-locomotion:T041AB`、git 2149c4f） |
| `TASK-036/ab/{exp_A_baseline,exp_B_nostand}/{log,err.log}` | 两组训练日志（`EXIT_train_*=0`） |
| `TASK-036/ab/eval_*` | 两组评估产物（`EXIT_eval_*=0`，各 4 场景 120 回合） |
| `TASK-036/ab/smoke_nostand` 的 `params/env.yaml` | 钩子生效的直接证据 |

## 42. 延长训练：第 41 章的乐观判断被推翻（TASK-036 裁决② 续）

第 41 章看到"只续训 250 迭代就把 −0.30 从 0.1031 压到 0.0602"，判断训练不足是主因。
这一章沿同一条轨迹再训 500 迭代（3248 → 3747），在 3500 和 3747 各评一次。**结论要改**。

### 42.1 结果

| 迭代 | +0.15 | −0.15 | +0.30 | −0.30 |
| --- | --- | --- | --- | --- |
| 2999（交付模型） | 0.1497 | 0.1501 | 0.0888 | **0.1031** |
| 3248（A 基线，第 41 章） | 0.1025 | 0.1497 | 0.0722 | 0.0602 |
| 3500（本轮中点） | 0.1443 | 0.1495 | 0.0716 | 0.0762 |
| 3747（本轮终点） | 0.1467 | 0.1471 | 0.0965 | 0.0666 |
| 3248（B 去站立，对照） | **0.0889** | **0.0810** | **0.0633** | **0.0575** |

（每格为 rmse，单种子 100、30 回合、4 个扫描点；后三行都从 2999 出发）

### 42.2 三件事

**一、3248 是个谷，不是趋势。** +0.15 在 3248 是 0.1025，到 3500 回到 0.1443、3747 是 0.1467，
基本等于交付模型的 0.1497。+0.30 同样：0.0722 → 0.0716 → 0.0965。
第 41 章"继续训就好"的判断建立在这一个点上，**站不住**。

**二、死区在整个 2999–3747 区间没有动过。** ±0.15 始终在 0.147–0.150，
实测速度一直只有 0.005–0.011（命令的 3–7 %）。900 多个迭代、两种调度，
**低速死区一次都没有被训练消除**。

**三、稳定的改善只有 −0.30。** 0.1031 → 0.0602/0.0762/0.0666，三个点都明显低于交付模型。
也就是说继续训练确实把 TASK-036 那个不达标项压下去了，但代价是 +0.30 从 0.0722 退到 0.0965
（仍在 0.10 以内，只剩 3.5 % 余量）。

### 42.3 修正后的判断

| 结论 | 第 41 章 | 现在 |
| --- | --- | --- |
| 训练不足是 TASK-036 失败主因 | 是 | **部分是，但不充分** |
| 继续训练能解决 lateral | 能 | **能压低 −0.30，但 +0.30 退回边缘，死区不动** |
| 改采样（去站立）是必要条件 | 锦上添花 | **是唯一让 ±0.15 动起来的手段**（0.0810 vs 0.147+） |

用交付标准衡量：3747 的 ±0.30 是 0.0965 / 0.0666，**都能过 0.10**，但 +0.30 只剩 3.5 % 余量。
考虑到这还只是单种子，换到 10 种子几乎必然有翻车风险——**不足以支撑"重训后就能通过"**。

而 B 组（去站立采样）在同样的 3248 上四个点全部明显更好（0.0889/0.0810/0.0633/0.0575），
**这才是有效方向**，而且它同时改善了死区。

### 42.4 方法论教训

单点评估会骗人。第 41 章只看了 3248 一个点就下了"训练没收敛"的判断，
500 迭代后同一配置退回去了。正确做法是**多点 + 多种子**——这也正是 TASK-036 用
10 个种子 × 每变体 300 回合、而不是抽一次就下结论的原因。
本轮 A/B 与延长实验都只有单种子 30 回合，只能用来判断"方向"，不能用来定"能不能过"。

### 42.5 下一步的两个选项

1. **沿 B 口径延长**（从 B 的 3248 再训 500，看四个点是否一起保持/改善）。成本约 40 分钟。
   若 B 的四个点继续保持在 0.06–0.09 且不退回，才值得投完整重训。
2. **直接按 B 口径重训**（从 2999 训到 5000+，中途多点评估选模），成本 3–4 小时，
   然后走选模 → 导出 → 闭环 → 10 种子复测。

在拿到多点、多种子的证据之前，不更新 TASK-036 的结论。

### 42.6 产物

| 文件 | 内容 |
| --- | --- |
| `/pub/data/caohy/scripts/t042_longer.sh` | 驱动：从 3248 续训 500 → 评 3500 与 3747 |
| `TASK-036/longer/longer_run.log` | 现场日志（窗口 `camra-locomotion:T042LONG`、EXIT 全 0） |
| `TASK-036/longer/exp_A_longer/` | 训练 run（含 model_3300…3747） |
| `TASK-036/longer/eval_{mid,last}/` | 两次评估产物（4 场景 × 30 回合） |

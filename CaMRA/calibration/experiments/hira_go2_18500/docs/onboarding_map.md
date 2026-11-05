# HiRA 接入接手地图（onboarding map）

> 面向"新机器 + 新上下文"的执行者。只凭本文件与所列路径即可定位入口、输入输出与需配置的资源。
> 生成于 HIRA-005（2026-11-05），依据 HIRA-001..004 的证据。
> **当前状态：尚未接入通过。** 缺环境、缺权限或缺资源，一律算环境问题，不得当成策略性能失败。

---

## 1. 全部入口（按阅读顺序）

| # | 入口 | 作用 | 绝对路径 |
|---|---|---|---|
| 1 | 根 AGENTS.md | 硬约束、执行与交接规则、论文只读令 | `/home/caohy/repositories/Paper/AGENTS.md` |
| 2 | 本轮计划（40 项） | 任务、依赖、验收、逐项记录 | `/home/caohy/repositories/Paper/doc/plan/20261105_camra_hira_policy_calibration.json` |
| 3 | 本轮 handoff | 基线、风险、方法、实现地图、源码语义 | `.../CaMRA/calibration/experiments/hira_go2_18500/docs/implementation_handoff.md` |
| 4 | 本接手地图 | 入口、配置对照、数据流、资源缺口 | `.../experiments/hira_go2_18500/docs/onboarding_map.md` |
| 5 | 就绪缺口（机器可读） | 四类未知项与责任任务 | `.../experiments/hira_go2_18500/reports/readiness_gaps.json` |
| 6 | 基线报告 | 版本、设备、GPU 归属、资源 | `.../reports/repository_baseline.json` |
| 7 | 现有实现地图（20 阶段） | 文件/函数/输入/输出 | `.../reports/existing_pipeline_map.json` |
| 8 | 来源语义追踪 | 45 维/动作/PD/时钟/命令改写 | `.../reports/hira_source_trace.json` |
| 9 | v2 历史证据 | §1–§32，只是历史，不作新数据 | `.../CaMRA/calibration/docs/implementation_handoff.md` |
| 10 | 资源获取说明 | 冻结包/检查点/资产/日志的取用与校验 | `.../CaMRA/locomotion/docs/reproduction.md` |

---

## 2. 来源 → 目标配置对照（接入必须逐项对齐）

| 项 | HiRA 来源（Isaac Gym） | 本轮目标（Isaac Lab） | 责任任务 |
|---|---|---|---|
| 观测维度 | 45（六段） | 45，顺序与缩放必须一致 | HIRA-013 |
| 观测归一化 | 无 | 无（不得擅自加） | HIRA-013 |
| 推理方式 | 均值（`act_inference`） | 均值，不采样 | HIRA-011 |
| 动作维度 | 12 | 12 | HIRA-014 |
| 动作缩放 | **0.25**（Go2 覆盖基类 0.5） | 0.25，显式断言 | HIRA-008/014 |
| 动作裁剪 | `clip(±100)`，在 step 入口一次 | 同样一次，不重复裁剪 | HIRA-014 |
| PD | P 控制，kp=20，kd=0.5，**每物理子步重算** | 每子步重算，不可简化为每策略步一次 | HIRA-014 |
| 力矩限幅 | URDF `effort` 逐关节 | Lab 导入后的 motor effort，需对照 | HIRA-012 |
| 物理步长 | 0.005 s | 0.005 s | HIRA-016 |
| 策略周期 | 0.02 s（decimation 4） | 0.02 s | HIRA-016 |
| 外部命令 | 0.1 s（= 5 策略步） | 0.1 s | HIRA-016 |
| 内部命令重采样 | 10 s，且 **wz 每步被 heading 覆盖** | 必须 `heading_command=False` | HIRA-016 |
| 低速死区 | 平面速度模长 ≤0.2 m/s 清零 | 必须关闭或显式记录 | HIRA-016 |
| 初始姿态 | `q_default × rand(0.5,1.5)`，z=1 m 落下 | 固定姿态、固定高度 | HIRA-012/016 |
| episode 长度 | 20 s | 采集窗口 30 s（协议），需说明差异 | HIRA-016/023 |
| 关节顺序 | 源 DOF 顺序（由资产构造决定） | 按名称建立双向置换 | HIRA-013 |
| 四元数约定 | 源用 `quat_apply`（xyzw 常见） | Lab 常用 wxyz，需逐项确认 | HIRA-013 |
| 关节默认角 | 12 个具名值（见 `source_default_angles_by_name`） | 按名称映射，不按顺序 | HIRA-013 |

---

## 3. 数据流：从速度命令到校准结果

```
[1] 外部命令 [vx,vy,wz] @10 Hz
        │  写入 commands[:, :3]；必须保证不在 5 个策略步内被改写
        ▼
[2] 观测拼装（45 维）：角速度×0.25 | 投影重力 | 命令×[2,2,0.25] | q−q_default | q̇×0.05 | 上一动作
        │  go2_env.compute_observations
        ▼
[3] actor 前向（均值）→ 12 维动作 → clip(±100)
        │  rsl_rl ActorCritic.act_inference
        ▼
[4] 关节目标 q_target = 0.25·a + q_default
        ▼
[5] 每物理子步：τ = clip(kp·(q_target−q) − kd·q̇, ±effort) → 送入物理
        │  legged_robot._compute_torques，重复 4 次
        ▼
[6] 记录（200 Hz）：τ、q̇、root 位姿/速度 → 逐子步能量
        │  sim/recorder.TrialRecorder（本轮需为 HiRA 重接）
        ▼
[7] 能量 J_exec = Σ_k Σ_j max(τ_j·q̇_j, 0)·0.005
        ▼
[8] 名义轨迹：同一命令序列做平面积分 → J_nom = Σ P̂·0.1 s
        │  nominal.integrate + 功率模型
        ▼
[9] 配对误差：ε_c = (J_exec−J_nom)/J_nom；ε_d = max_t‖p_exec−p_nom‖
        │  pairing.compute_trial_errors
        ▼
[10] 裕度：α = Q0.9(ε_c)、δ = Q0.9(ε_d)（nearest-rank）→ 冻结 → 独立检验
```

**注意**：[8] 的名义轨迹只能用命令积分，**不能用 clean rollout 顶替**；clean rollout 仍属执行系统。

---

## 4. 资源清单与取得路径（本机实测，2026-11-05）

| 资源 | 本机路径 | 状态 | 取得方式 |
|---|---|---|---|
| 环境 `env_isaaclab` | `/pub/data/caohy/miniconda/envs/env_isaaclab` | ✅ 在位 | 已存在，torch 2.7.0+cu126 / isaaclab 0.47.2 |
| HiRA 子模块（含 actor、URDF） | `Paper/HiRA` @ `6a05af55` | ✅ 在位 | 仓库内，只读；未拉新提交 |
| HiRA actor checkpoint | `HiRA/logs/rough_go2/Sep08_11-57-26_/model_18500.pt` | ✅ 在位（4.4 MB） | 子模块自带，sha `d30e5212…` |
| Go2 URDF | `HiRA/legged_gym_go2/resources/robots/go2/urdf/go2.urdf` | ✅ 在位（25 KB） | 子模块自带 |
| Go2 USD | `/pub/data/caohy/unitree_model/Go2/usd` | ✅ 在位（18 MB） | 本机资产根 |
| v2 冻结包 | `/pub/data/caohy/camra_bundles/frozen/camra_go2_velocity_it2999_v2` | ✅ 在位（28 MB） | 只作对照，不改 |
| v2 训练目录 | `/pub/data/caohy/logs/rsl_rl/rsl_rl/camra_go2_locomotion` | ✅ 在位（4.5 MB） | 只读 |
| **v2 校准原始产物**（1820 条） | `/pub/data/caohy/logs/camra/trials` | ✅ 在位（68 GB） | 只读；HIRA-037 对照用 |
| 候选包 `r4_it2999_v1` | `/pub/data/caohy/camra_bundles/candidates/` | ❌ **本机缺失** | 旧设备 `caohy@192.168.20.12` 同名路径 rsync |
| v2 底层评估原始日志 TASK-024 / 026 / 036 | `/pub/data/caohy/logs/camra/TASK-0xx` | ❌ **本机缺失**（662 MB / 10.6 MB / 1.3 GB） | 旧设备 `.12` 同名路径 rsync |
| 本轮实验根 | `.../experiments/hira_go2_18500/{config,reports,docs,tests}` | ✅ 已建 | — |
| 本轮日志根 | `/pub/data/caohy/logs/camra_hira/{env_setup,integration,trials}` | ✅ 已建 | 与 v2 隔离 |

**权限**：本机 `caohy` 对上述 `/pub/data/caohy/**` 与仓库均有读写；旧设备 `.12` 的访问需 ssh 内网，
`reproduction.md` 给出 rsync 命令。**没有任何凭证入库。**

> 本机缺失项不阻塞接入验证（HIRA-006..020 只需 HiRA 权重 + 资产 + 环境），
> 但会影响 HIRA-037 的"与 v2 同口径比较"深度：若拿不到 TASK-024/036 原始日志，
> 只能用 v2 冻结报告 + 本机 1820 条 trial 产物做对照，**这一点必须届时如实说明**。

---

## 5. 四类未知项 → 责任任务

### 5.1 来源不明（无法从代码或 checkpoint 恢复）

| 项 | 为什么不明 | 责任任务 |
|---|---|---|
| 训练期命令课程范围 | checkpoint `infos=None` | HIRA-009（预登记时声明为不可得） |
| domain randomization 参数 | 同上 | HIRA-008（契约里标为未知） |
| 观测噪声配置与随机种子 | 同上 | HIRA-013（clean 关闭噪声，扰动按新 profile） |
| 训练步数 | 文件名 18500 vs `iter=16000` 不一致 | HIRA-011（以 actor 权重身份为准，不写步数） |
| 训练期是否覆盖 PD/scale | 覆盖不可验证 | HIRA-014（以源码值 20/0.5/0.25 为准并断言） |

### 5.2 可测映射（能用数值对照确定）

| 项 | 怎么测 | 责任任务 |
|---|---|---|
| 关节顺序置换 | 按 DOF 名称建立双向表，再用同姿态力矩对照 | HIRA-013 |
| 默认角向量 | 按名称映射后与源 12 个具名值比对 | HIRA-013 |
| 四元数约定 | 用同姿态算投影重力，比较符号/分量 | HIRA-013 |
| 观测裁剪是否启用 | 构造越界观测看是否被截 | HIRA-013 |
| URDF effort vs Lab effort | 读两侧限幅值逐关节对照 | HIRA-012 |
| 高层 repeat 相位 | 固定命令 5 步，检查命令不变 | HIRA-016 |

### 5.3 动力学差异（不可完全消除，只能量化）

| 项 | 差异 | 责任任务 |
|---|---|---|
| 执行器模型 | Gym 手动 set force vs Lab actuator 模型 | HIRA-012/014 |
| 接触与摩擦 | 两引擎求解器不同 | HIRA-017（闭环行为对照，不要求逐帧一致） |
| 资产 | URDF（源）vs USD（Lab） | HIRA-012 |
| 结论口径 | 只能称"HiRA 策略在本次 Isaac Lab 执行系统中的结果" | HIRA-020（冻结包措辞） |

### 5.4 旧工具绑定（8 处默认值必须覆盖）

| 绑定 | 值 | 责任任务 |
|---|---|---|
| `collect_trials --device` / `CAMRA_DEVICE` | `cuda:0`（GPU0 有他人 katago） | HIRA-021 |
| `CAMRA_LOG_ROOT` | `/pub/data/caohy/logs/camra`（v2 树） | HIRA-021 |
| `CAMRA_TMUX_SESSION` | `camra-locomotion` | HIRA-021 |
| `CONTROLLER_VERSION` | `camra_go2_velocity_it2999_v2` | HIRA-010 |
| `protocol.DEFAULT_VERSION` | `1.0.2` | HIRA-023 |
| `lock.enforce_collection` 默认 margins | `config/margins_v1.json` | HIRA-029 |
| `build_data_lock` 代码树范围 | `rglob('*.py')` 覆盖 v2 全部 54 个 | HIRA-029 |
| `--allow-overwrite` | 正式 split 一律 REFUSED（保护） | HIRA-021（沿用） |

---

## 6. 当前就绪判断

| 维度 | 状态 | 说明 |
|---|---|---|
| 环境 | 就绪 | env_isaaclab 在位，GPU2 空闲 |
| 权重 | 就绪（身份已核） | sha 与计划一致；actor 张量形状已核对 |
| 资产 | 就绪 | URDF + USD 均在位 |
| 源码可读性 | 就绪 | 子模块干净，语义已追踪 |
| 接入实现 | **未开始** | HIRA-010 起才写适配器 |
| G1 门禁 | **未通过** | 未做任何数值对照或闭环 |
| 旧日志完整性 | 部分缺失 | TASK-024/026/036 本机无，影响 HIRA-037 深度 |

**结论：可以进入 HIRA-006（配置匹配环境与独立路径），但不得开始任何采集或宣称接入成功。**

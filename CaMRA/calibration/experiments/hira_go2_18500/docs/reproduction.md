# HiRA 策略校准：复现说明（HIRA-038）

本文是交付包的操作手册。**先读 §1 的证据等级**，再按 §3 复现；§6 是"结论在哪一份报告的哪个字段"，
§7 是必须连同结论一起读的边界。

**2026-11-09 文档复核更新**：40项计划已记录完成，tiger11 数据迁移已报告通过；当前新增的采集初态、名义起点对齐、全体分母和交付索引问题见[复核说明](../../../docs/controller_comparison_status_20261109.md)。本文同步当前入口，不覆盖历史报告、冻结值或锁；原交付索引记录的是历史字节快照，本文后续修订不通过重算旧索引掩盖。下列期望值用于复现历史计算，不表示新增问题已解决。

当前主线已固定使用 v2，并准备执行[60项导航计划](../../../../../doc/plan/20261109_camra_navigation_implementation.json)（1.2.0，0/60完成）。本文仅供历史 HiRA 实验追溯，不授权重新采集或将其控制器、路径、锁用于 v2 导航；新上下文先读[根 AGENTS](../../../../../AGENTS.md)。

---

## 1. 这个包是什么，证据等级是什么

**是什么**：把 HiRA（Isaac Gym 训练）搬进 Isaac Lab 后，重新量出导航层需要的两个余量
——能量系数 α 与位置偏差 δ——并给出它们在新条件下的有效性检验。

**冻结结论**

| 量 | 值 | 出处 |
|---|---|---|
| α（能量放大系数） | **0.35156762163783695** | `config/margins_hira_v1.json` |
| δ（最大位置误差的经验90%分位数） | **9.032231244495946 m** | 同上 |
| 计算分母 | 771 条完整 calibration（800 - 29 业务失败） | `reports/margin_calibration.json` |

**证据等级（三条，必须一起读）**

1. **HiRA 侧是可复算的**：1820 条原始记录 + 独立实现审计（`reports/full_independent_audit.json`，
   12/12，能量最差相对误差 4.97e-16，α/δ 逐位一致）。
2. **v2 侧只有报告级证据**：v2 的 68 GB 逐 trial 数据已不在本机，`reports/hira_vs_v2_comparison.json`
   的 `recomputed=false`。那张表是**描述性对照**，不是配对检验，不能当因果。
3. **近障结论是负结果**：δ = 9.03 m 已接近 8×10 m 场地尺度，200 条里 **0 条**满足 δ 条件。
   这不是采集失败，是这套余量在该场地尺度上不可用（§6.3）。

---

## 2. 包里有什么

```
<DELIVERY_ROOT>/
└── CaMRA/
    ├── locomotion/adapters/hira_go2/          # 13 个 .py：适配层（含 obstacles.py）
    └── calibration/
        ├── src/camra_calibration/             # 24 个 .py：锁/清单/能量/裕度/覆盖/几何
        └── experiments/hira_go2_18500/
            ├── config/                        # 契约、协议、预登记、功率、轮廓、裕度、4 把锁
            ├── frozen/hira_go2_18500/         # 控制器包（bundle_id 3c32fe31…）
            ├── manifests/                     # 七个 split 的 1820 条清单 + 清单锁
            ├── tests/                         # 检查/评估/审计/交付入口脚本
            ├── reports/                       # 全部判决性报告（§6 的索引）
            ├── docs/                          # implementation_handoff.md + 本文
            └── delivery/bundle_manifest_hira_v1.json   # 本包索引（逐文件 SHA256）
```

**不在包里**（`bundle_manifest_hira_v1.json` 的 `big_data` / `not_bundled` 段有登记）：

| 对象 | 体积 | 取用方式 |
|---|---|---|
| 1880 个索引 trial 原始记录（1820 正式配额及同根辅助记录） | ≈ 2.1 GB | tiger11 `/pub/data/caohy/logs/camra_hira/runs`；迁移及散列证据见 handoff §45 |
| USD 缓存（由 URDF 转换得到） | 18 MB | 可由 `config` 的资产流程重建，非唯一证据 |
| 稿件 `manuscripts/**`、`*.tex`、`*.bib` | — | **不打包**：红线，改动清单见 handoff §41 |
| 凭证、`fakehome`、机器私密配置 | — | **不打包** |

---

## 3. 复现（换机 / 换目录）

### 3.0 为什么"换个目录就能跑"

所有脚本用 `__file__` 自定位：`EXP = tests/../`，`REPO = EXP` 上四级。
把 §2 的三棵树按**同样的相对布局**放到新根，源码零改动即可运行。

**唯一需要重定位的是 stage lock 里的绝对路径**（`code.files` 的键、`controller.*`、`protocol`、
`margins`、`trial_lists`、`gates`、`near_obstacle_maps` 全是 `/pub/...` 绝对路径）。
`tests/delivery_relock.py` 生成派生锁：只改路径前缀，**所有 sha256 逐字保留**，原锁不动。

### 3.1 前置

- **执行主机**：按 `AGENTS.md` 的现行要求为 `192.168.20.11`（tiger11；2026-11-09 用户更正，初版误写为 `.10`，`.10` 不使用）。
  历史数据产生于 `tiger12`，两台的 `/pub` 是各自本地盘；交接笔记 §45 和 `reports/migration_tiger11_20261109.json` 已记录工作树、9 个 run、1880 个索引 trial 与7个 USD 缓存文件迁移校验通过，源端保留。运行前再核验路径、资源及环境，不默认重搬；此次本地文档同步未重新连接服务器。
- Python 3.11 + numpy + torch（重载与离线复算够用）
- 采集另需 Isaac Lab 环境（执行机 `/pub/data/caohy/miniconda/envs/env_isaaclab`）；tiger11 为 Python 3.11.15，历史 tiger12 为3.11.18，仍须核验 GPU 和运行依赖
- 参考 `config/hira.env`：换机器**只改这一个文件**里的根路径

### 3.2 逐项校验

```bash
cd <DELIVERY_ROOT>/CaMRA/calibration/experiments/hira_go2_18500

# ① 按历史索引逐文件（含1880个trial的树散列）复核；原交付快照期望 bad=0
# 当前 Git 检出另有缺失日志、换行及索引时序问题，且文档已后续修订，见本文开头链接
python tests/build_delivery_manifest.py --verify delivery/bundle_manifest_hira_v1.json

# ② 生成派生锁（只改路径前缀）；期望 verify_ok=true、n_paths_rewritten=57
#    当前最新是检验锁 v3（d7d60cb9，含 run_split 的输出目录绑定守卫）；
#    v1（289e4137）仍是 heldout 400 条的归属凭证、v2（db156f4d）仍是 near 200 条的，
#    复核历史数据时请用对应版本，不要用 v3 覆盖它们。
python tests/delivery_relock.py \
  --lock config/data_lock_hira_evaluation_v3.json \
  --out  config/data_lock_hira_evaluation_delivery.json \
  --new-root <DELIVERY_ROOT> \
  --old-prefix /pub/data/caohy/work/hira/
```

### 3.3 一条命令跑完 20 项重载验证

```bash
python tests/verify_delivery_reload.py \
  --work-root <NEW_SCRATCH_ROOT> \
  --run-id delivery_reload_$(date +%H%M)
# 期望：=== 20/20 passed, failed=[] ===
# 产物：reports/delivery_reload_verification.json
```

它做的事（对应 `checks[].id`）：D01 独立根 → D03 三树逐文件散列一致 → D05/D06 派生锁在新根
`verify_artifacts` 全过、D07 原锁未被改 → D08 模型被换拒绝 → D09 权重缺失明确报错 →
D10 地图被换门禁拒绝 → D11 资产缺失 preflight 报缺失 → D12 输出目录误换交付入口拒绝 →
D13 bundle 重载 → D14 obs/动作 parity → D15 小批采集 → D16 官方目录零改动 →
D17 障碍注入证据 → D18 ε_c/ε_d 与覆盖 → D20 报告可定位 → D19 离线能量复算。

### 3.4 正式采集的唯一入口

```bash
python tests/run_delivery.py --split near_obstacle --run-id <NEW_ID> \
  --lock config/data_lock_hira_evaluation_delivery.json \
  --maps config/near_obstacle_maps_v1.json --num-envs 8
```

`run_delivery.py` 和当前 `run_split.py` 均核验目标 run 目录绑定的 split / 锁散列 / bundle / 清单，不符即
**exit 2 拒绝启动**。该守卫已纳入检验锁 v3；历史 v1/v2 的数据归属不变。`run_delivery.py --preflight-only` 可只核验。

### 3.5 离线复算（不需要 GPU / Isaac）

```bash
# 能量：从 physics_200hz.npz 的 tau_applied_source × qdot_post_source 重积分
#       与 energy.json 的 j_exec 比较，门槛 rel <= 1e-9（实测 4.97e-16）
# 覆盖：用冻结 α/δ 重算 eps_c = (J_exec-J_nom)/J_nom、eps_d = max|exec-nominal|，
#       与 reports/heldout_coverage.json、reports/near_obstacle_evaluation.json 对账
python tests/full_independent_audit.py     # 12/12，含上述复算
```

---

## 4. 口径速查（算错就会得到不同的数）

| 项 | 规定 |
|---|---|
| 观测 | 45 维：`ang_vel×0.25 | gravity×1 | command×[2,2,0.25] | dof_pos | dof_vel×0.05 | prev_action`；噪声只落 9:33 |
| 力矩 | `τ = clip(kp(q_def+0.25a−q) − kd·q̇, ±effort)`；kp=20、kd=0.5；**effort 逐关节**（calf 35.55，其余 23.7） |
| 时钟 | 策略 50 Hz、命令 10 Hz、物理 200 Hz（每策略步 4 子步，每子步重算 PD） |
| 能量 | `J = ΣΣmax(τ·q̇,0)·0.005`，逐关节先截断再求和，τ 用 `applied_torque`、q̇ 用**子步结束**时刻 |
| ε_c | `(J_exec − J_nom)/J_nom`，`J_nom < 1 J` 记 non_computable |
| ε_d | 10 Hz 配对网格上执行与名义位置的**最大平面欧氏距离**（名义由 `build_program` + 零阶保持积分重建） |
| 覆盖 | 等于阈值算覆盖；联合覆盖单独报，不用边际乘积顶替 |
| 区间 | cell 整簇 bootstrap 10000×95%；重采样单位是整条轨迹，绝不按时间点伪增 n |

---

## 5. 数字对不对：期望值清单

| 量 | 期望 | 容差 |
|---|---|---|
| α | 0.35156762163783695 | 逐位 |
| δ | 9.032231244495946 m | 逐位 |
| 能量（1820 条） | 与 `energy.json` 一致 | 最差 rel ≤ 1e-9（实测 4.97e-16） |
| heldout 覆盖（n=377） | 成本 0.9098 / 位置 0.8966 / 联合 0.8170 | 逐位 |
| 近障覆盖（n=188） | 0.9362 / 0.9840 / 0.9202 | 逐位 |
| 近障几何重叠 | 151/200 | 逐条 |
| 近障 δ 条件 | **0/200**（两种口径都是 0） | 逐条 |
| 独立审计 | 12/12，v2 未改动 | — |

表中覆盖率均为完整且可计算子集的条件覆盖率。由保存覆盖计数除以全体预登记配额，heldout 成本/位置/联合为343/400（85.75%）、338/400（84.50%）、308/400（77.00%）；近障为176/200（88.00%）、185/200（92.50%）、173/200（86.50%）。未完成试验不计已覆盖但保留在分母；这项算术补充未重算逐轨迹误差和区间，也未覆盖原报告。

---

## 6. 结论在哪一份报告的哪个字段

### 6.1 校准与冻结
- `reports/margin_calibration.json` — α/δ 推导、分支区间、敏感性
- `config/margins_hira_v1.json` — 冻结值（唯一权威副本，含 `sha256` 与 `content_sha256`）

### 6.2 独立检验（heldout，400 条）
- `reports/heldout_collection_integrity.json` — C01–C10；`counts`：377 完成 / 23 业务失败 / 0 排除
- `reports/heldout_coverage.json` — `full_sample`、`gap_cells_removed`、`per_cell`、`verdict`
- `reports/heldout_summary.md` — 人读版

**未达标项（如实记录，未修正）**：条件位置覆盖 0.8966，比 90% 参照低 **0.34 pp**；条件联合覆盖 0.8170 低于90%。联合/(成本×位置) = 1.0016 仅说明该样本中事件频率接近乘积，不能据此证明两项误差统计独立。现有 `full_sample.overall` 的分母为377，不是全部400条；全体口径见§5及最新复核说明。

### 6.3 近障（200 条）
- `reports/near_obstacle_collection_integrity.json` — N01–N11，含障碍注入证据（1338 个障碍、
  位姿读回偏差 4.76e-07 m）
- `reports/near_obstacle_evaluation.json` — `geometry.executed`（重叠 151、越界 15）、
  `geometry.delta_condition`（0/200）、`geometry.delta_condition.robustness_centerline`（0/200）
- `reports/near_obstacle_summary.md` — 人读版

### 6.4 审计与对照
- `reports/full_independent_audit.json` — A01–A12（独立实现，不 import 被测模块）
- `reports/audit_discrepancies.json` — 2 条已解释差异
- `reports/hira_vs_v2_comparison.md` — 与 v2 的报告级对照（10 项可比性 + 14 项指标）

### 6.5 交付验证
- `reports/delivery_reload_verification.json` — D01–D20、`verdict.reload_ok`
- `delivery/bundle_manifest_hira_v1.json` — 逐文件散列、大数据索引、外部依赖与"不打包"清单

---

## 7. 边界：必须连同结论一起读

1. **近障 0/200 不是"避障失败率 100%"**。控制器只是按冻结命令回放，没有上层导航在避障；
   它测的是"这套余量在该场地能不能用"，答案是不能。
2. **位置覆盖高 ≠ 安全**。近障条件位置覆盖 98.40%，同时全体几何重叠 75.5%：δ 是**最大跟踪误差的经验分位数**，
   不是碰撞裕度；危险集扩张 δ 是必要条件，不是充分条件。
3. **格级结论只有 17/20（heldout）、18/20（近障）**。三格/两格完整率低于 0.9，按裁定保留在主分母
   但不进格级结论；不得宣称"全部格达标"。
4. **warmup 有 1 s 零动作默认角 PD 前置**，并非调用策略跟踪零速度命令；普通批次起点的完整关节/上一动作复位，以及 warmup 后的名义起点实测对齐仍待复核。历史数据和旧锁保留，必要修复另出版本，不悄悄改原采集口径。
5. **输出目录守卫已覆盖两入口**：当前 `run_split.py` 与 `run_delivery.py` 都有检查，检验锁v3已绑定该版本。重放历史数据用其归属锁及相应代码快照，不将当前代码塞回旧锁。
6. **与 v2 的数字不可混用**：控制器包同时换了权重、默认姿态、kp（25→20）与机器人资产；
   v2 还有两次整批重采，HiRA 是零重采口径。任何单指标差异都不能归因于网络权重一项。
7. **物理接触未记录接触力**：`n_overlap=151` 是几何相交（碰撞下界证据），不是接触冲量。

---

## 8. 复现失败时先查这三处

| 症状 | 原因 | 处置 |
|---|---|---|
| `verify_artifacts` 报一堆"文件不存在" | 用了**原锁**而不是派生锁（原锁是 `/pub` 绝对路径） | 重跑 §3.2 ② 生成派生锁 |
| `enforce_collection` 报 code 散列不符 | 副本里的代码被改过，或索引与代码不同版本 | `--verify` 逐文件比对，定位改动 |
| 采集启动即报 `--maps 散列与锁记录不一致` | 地图被换或换了别的 split | 用 `config/near_obstacle_maps_v1.json` 原件（`sha256 78c0145f…`） |

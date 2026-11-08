# HiRA 策略校准：复现说明（HIRA-038）

本文是交付包的操作手册。**先读 §1 的证据等级**，再按 §3 复现；§6 是"结论在哪一份报告的哪个字段"，
§7 是必须连同结论一起读的边界。

---

## 1. 这个包是什么，证据等级是什么

**是什么**：把 HiRA（Isaac Gym 训练）搬进 Isaac Lab 后，重新量出导航层需要的两个余量
——能量系数 α 与位置偏差 δ——并给出它们在新条件下的有效性检验。

**冻结结论**

| 量 | 值 | 出处 |
|---|---|---|
| α（能量放大系数） | **0.35156762163783695** | `config/margins_hira_v1.json` |
| δ（位置偏差上界） | **9.032231244495946 m** | 同上 |
| 计算分母 | 771 条完整 calibration（800 - 29 业务失败） | `reports/margin_calibration.json` |

**证据等级（三条，必须一起读）**

1. **HiRA 侧是可复算的**：1820 条原始记录 + 独立实现审计（`reports/full_independent_audit.json`，
   12/12，能量最差相对误差 4.97e-16，α/δ 逐位一致）。
2. **v2 侧只有报告级证据**：v2 的 68 GB 逐 trial 数据已不在本机，`reports/hira_vs_v2_comparison.json`
   的 `recomputed=false`。那张表是**描述性对照**，不是配对检验，不能当因果。
3. **近障结论是负结果**：δ = 9.03 m 与 8×10 m 场地差一个数量级，200 条里 **0 条**满足 δ 条件。
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
| 1830 个 trial 原始记录（`physics_200hz.npz` 等） | ≈ 2.1 GB | 挂到 `/pub/data/caohy/logs/camra_hira/runs`，或改写 `--runs-root` |
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

- Python 3.11 + numpy + torch（重载与离线复算够用）
- 采集另需 Isaac Lab 环境（本机 `/pub/data/caohy/miniconda/envs/env_isaaclab`）
- 参考 `config/hira.env`：换机器**只改这一个文件**里的根路径

### 3.2 逐项校验

```bash
cd <DELIVERY_ROOT>/CaMRA/calibration/experiments/hira_go2_18500

# ① 按索引逐文件（含 1830 个 trial 的树散列）复核；期望 bad=0
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

`run_delivery.py` 先核验目标 run 目录绑定的 split / 锁散列 / bundle / 清单，不符即
**exit 2 拒绝启动**（`run_split.py` 本身没有这道核验，见 §7.5）。`--preflight-only` 可只核验。

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

---

## 6. 结论在哪一份报告的哪个字段

### 6.1 校准与冻结
- `reports/margin_calibration.json` — α/δ 推导、分支区间、敏感性
- `config/margins_hira_v1.json` — 冻结值（唯一权威副本，含 `sha256` 与 `content_sha256`）

### 6.2 独立检验（heldout，400 条）
- `reports/heldout_collection_integrity.json` — C01–C10；`counts`：377 完成 / 23 业务失败 / 0 排除
- `reports/heldout_coverage.json` — `full_sample`、`gap_cells_removed`、`per_cell`、`verdict`
- `reports/heldout_summary.md` — 人读版

**未达标项（如实记录，未修正）**：位置覆盖 0.8966，比 90% 参照低 **0.34 pp**；联合 0.8170 低于 90%，
但 联合/(成本×位置) = 1.0016，说明两项误差独立、联合缺口是边际的必然结果。

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
2. **位置覆盖高 ≠ 安全**。近障位置覆盖 98.40%，同时几何重叠 75.5%：δ 是**跟踪误差上界**，
   不是碰撞裕度；危险集扩张 δ 是必要条件，不是充分条件。
3. **格级结论只有 17/20（heldout）、18/20（近障）**。三格/两格完整率低于 0.9，按裁定保留在主分母
   但不进格级结论；不得宣称"全部格达标"。
4. **warmup 有 1 s 零动作前置**（与 800 条校准同口径）。本版不修：修了与已采数据不可比。
5. **输出目录守卫在交付层**：`run_split.py` 启动时不校验既有 `collect_config`（该文件在锁内，
   未擅自改），改由 `run_delivery.py` 前置拦截。若要下沉到 `run_split` 本身，需新版锁。
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

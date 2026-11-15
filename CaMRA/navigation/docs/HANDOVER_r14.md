# NAV-064 第 14 轮（r14）交接文档

- 生成时间：2026-11-15 19:30（GMT+8）
- 生成背景：用户中断执行并要求交接；本文档替代口头上下文，任何人（或下一次会话）读它即可接手
- 一句话现状：**r14 的代码、契约、预登记、测试已全部完成并验证（32/32 绿），且没有启动训练。反例门禁两轮都失败——第一轮是数据集选错（无到达回合），第二轮查出真实缺陷：深井只进了 critic 与可行性判据，没有进策略优势（联合 min 递推在停止点取步前的 h_t）。按预登记，训练被门禁挡住；下一步需要一个决策（见第 6 节）。**

---

## 1. 这个项目在干什么（给完全不了解的人）

训练一个强化学习策略，让机器人在**限定电量**和**限定时间**内从起点走到终点，**全程不碰危险区**。三项要求是硬约束，不是加权打分。

- 判据（主指标）：`sr_feasible = 到达 ∧ 全程 safety_y=0 ∧ 结束电量 ≥ 0`，在 development 6 张图上、3 个种子、k=1.15 预算、T=45 s 时限下取中位数，门槛 **≥ 0.10**。
- 训练/评估数据分离：训练用 18 张图，判定只用 development 6 张，final 封存。
- 服务器：`tiger11`（192.168.20.11），训练根 `/pub/data/caohy/work/camra_navigation/`，仓库 `/home/caohy/repositories/Paper/CaMRA/navigation`（属于 `Paper` git 仓库，远程 `git@github.com:haoyangc2001/Paper.git`，主分支 `main`）。
- python 环境：`/pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python`（简写 `$PY`），必须带 `PYTHONPATH=src:../calibration/src`。

## 2. 为什么有第 14 轮

前 13 轮全部未达标。判负后的根因审计（工作记录第 38、39 条）定案：

1. **到达几乎无奖励**：`G_INSIDE = -0.001`，且 `h = max(g_norm, C·y)` 的 `max(·,0)` 把负值抹平——到达和原地不动的信号一样。
2. **加性成本罚压过前进信号**：w=50（超支步）、base=2（预算内步），而距离项每米只值 0.067；base=2 时前进每米净亏 0.022 → 策略学会“走六成就怠速”，与 r13 实测逐点吻合。
3. **观测饱和**：所有场景 `E_ref > Z_REF`，z_ratio 在 ±1.5 就被裁，成本通道对 critic 饱和。
4. 反证：r12 旧策略在 T=45/k=1.15 下拿到 SR 1.00（uncal_s0_nomask）——**运动能力本身存在**，13 轮不是白练，是每轮在优化一个写错的目标。

对照 RC-PPO 参考实现（`Paper/RC-PPO/rl/`）后确认：价值侧 `W=max(V_h, V_c−z)` 已对齐，**策略目标侧**偏离（增量罚 vs 参考的 max 结构 + min 型递推）。amend3 曾严格转写 min 型递推，因 h 被抹平导致 `corr(h,adv)` 在 3 个 run 里 2 个翻负而冻结（`joint_min_recursion=False`，训练器拒绝启用）。

## 3. r14 的授权与范围（用户 2026-11-15T17:32 三点决策）

1. **正式开工**，不再等；
2. **并入两个独立问题**：安全项连续化（F1/F2）+ 停滞规则放宽，并在对应位置补日志；
3. **最小电量搜索按推荐留给 NAV-066**（部署期模块，本轮不做）。

明确**不做**：部署预算搜索（M10）、final 集、任何墙钟上限（已由 `config/wallclock_amend_v1.json` 取消）。

## 4. 已完成的工作

### 4.1 契约升版（语义变更的正式登记）

| 文件 | 变更 | 说明 |
|---|---|---|
| `config/value_objective_contract_v1.json` | v1.2.0 → **v1.3.0** | `G_INSIDE` −0.001 → **−3.0**（深井）；新增 `quantities.safety_severity`（分级安全历史，floor=0.2、ref_depth=0.25）；符号约定里“ĝ 必为正”改为“ĝ 可为负”；`amendments` 追加 r14 条（含授权出处）；C 常数补 obs-v4 一致性说明 |
| `config/navigation_contract_v1.json` | v1.5.0 → **v1.6.0** | `observation_spec.version` 与顶层 `observation_version` → **obs-v4**；`z_ratio.range` ±1.5 → **±3.0**（布局/维度不变） |

**重要**：v1.3.0 语义下 r1–r13 的数值结果**不可直接比较**；唯一有效对照是 r13 的 B 臂（同契约、同 T/k，只差语义）。

### 4.2 代码改动（8 个文件，全部已上传服务器并通过测试）

| 文件 | 改动 |
|---|---|
| `src/camra_navigation/observation.py` | 新增 `obs-v4`（clip 3.0），`OBS_VERSION="obs-v4"`，`SUPPORTED_OBS_VERSIONS` 加 v4 |
| `src/camra_navigation/maps/geometry.py` | 新增 `update_safety_history_graded(y, severity)`＝不可逆 max-历史；新增 `safety_severity_from_depth(depth, floor, ref_depth)`＝`clip(floor+(1−floor)·d/ref, floor, 1)`。旧二值函数**逐字未动** |
| `src/camra_navigation/envs/nominal.py` | 新增三个构造参数（`safety_severity_enabled/floor/ref_depth_m`，默认关闭）；**裕度计算移到盖章之前**，违规步按侵深算 severity 再分级盖章；`StepRecord.safety_severity` 新字段；`EpisodeResult.max_safety_severity`/`stagnation_context`；`_stagnated()` 触发时捕获现场（位置/距目标/余电/窗口位移/剩余时间）；`trace()` 记新参数 |
| `src/camra_navigation/envs/vector.py` | 同上语义的向量版（参数透传、分级盖章、`BatchStepRecord.safety_severity`、停滞现场、`result()` 带两个诊断字段）；`reset()` 清空诊断状态 |
| `src/camra_navigation/algorithm/targets.py` | 新增 `safety_step_v2(g, y, C)`：`y>0 → max(g, C·y)`；`y=0 → g`（不裁 0）。旧 `safety_step` 未动 |
| `src/camra_navigation/algorithm/storage.py` | `ObjectiveConfig.h_semantics` 新字段（默认 `"v1"`）；`compute_objectives` 的 `h_t` 与 `h_T` 按 `v2_deep_well` 分支 |
| `scripts/train_formal.py` | ①读 `objective_signal_fix.h_semantics`（只允许 v1/v2_deep_well），v2 时校验收到的契约 `G_INSIDE ≤ −0.5`；②`joint_min_recursion=True` 现在**只在 v2 下放行**（v1 仍拒绝，amend3 反例仍有效）；③`build_envs` 从预登记的 `stagnation` / `safety_continuity` 读参数传给环境（含范围校验）；④`run_start` 日志记 `h_semantics`/`stagnation`/`safety_continuity`；⑤**每局 `episode_diag` 行**（label、步数、终点距离、余电、z0、max_safety_severity、danger_steps、within_budget、stagnation_context，在 env.reset 之前抓取）；⑥**每 update `win_diag`**（`corr_g_adv`、`corr_c_adv`、`h_min/h_mean`、`frac_h_below0`、`frac_overspend`、`adv_total_min/max`） |
| `scripts/contract_margin_eval.py` | 评估环境参数改为**读该 run 冻结的预登记**（停滞/分级安全），不再命令行另传；每局结果带 `max_safety_severity`/`stagnation_context`；每 run 汇总带 `env_kwargs`/`prereg` |

### 4.3 测试

- 新增 `tests/test_h_semantics_v2.py`（HS1–HS9，**20 项全过**）：v2 真值表、v1 逐位回归（只在目标集内不同）、分级历史不可逆、严重度公式、干净到达 h_T=−3、违规到达 h_T=C·y、到达优于差 1 m 停下的方向性、省钱放大深井、real env 分级盖章、停滞现场捕获、obs-v4 边界。
- 对齐 5 个旧测试的写死边界：`test_observation_contract`（lim 1.5→3.0、z 用例 ±1650）、`test_budget_selection`（饱和用例 e_ref 1500、比值 1950/550）、`test_rollout_resume`（饱和段 z_norm −3.2 / z_ratio −3.0、期望 g=3.2）、`test_training_semantics`（越界 z0 用例 e_ref 1400）、`test_value_objective`（C 支配性按分级语义重写：`C·floor > G_INSIDE`）。
- **顺手修掉两条陈年红项**（`test_nominal_env.py`：钉 obs-v2 的断言、NAV-062 后仍等 `budget_exhausted` 标签的断言），已对齐当前冻结语义并在注释里写明缘由。
- 结果：**`/tmp/run_all_tests.sh` → PASS=32 FAIL=0**。

### 4.4 预登记（已冻结）

- `config/training_loop_prereg_r14.json` + `.sha256`，**sha = `51a86ea3160e45dd89de899df59363f6916ca3f2bc6ea587a0f90dbdc4cb0b64`**
- 生成脚本：`scripts/t_build_r14_prereg.py`（以 r13 为底稿，带守卫断言：只允许 19 个顶层键变化，`environment/budget/algorithm/window_semantics/reference_energy/action_init_fix/steering_prior/dead_step_mask/action_bounded` 必须逐字相同）
- 关键字段：`objective_signal_fix = {w:0.0, base:0.0, min_recursion:true, cost_on_overspend_only:false, h_semantics:"v2_deep_well"}`；`cost_curriculum.enabled=false`；`stagnation={3.0s, 0.02m}`；`safety_continuity={on, 0.2, 0.25m}`；`reference_energy={t45 报告, k=1.15}`；`environment.t_max_s=45`；6 个 run（2 方法 × 3 种子，无臂）；绑定新值契约（sha `c0032c2a…`）、新导航契约（`332174c3…`）、r13 预登记/报告/负结果、`tests/test_h_semantics_v2.py`
- 证伪条件：(a) T4 门禁不过 → 回退方案 A；(b) dev `sr_feasible` 中位 < 0.10；(c) 日志 `corr_g_adv` update 级中位 ≤ 0 且训练期到达率低于 r13 同口径。

### 4.5 T4 反例门禁：两次尝试，都指向同一个第三方缺陷

- **第一版（无效，已归档 `reports/gate_r14_min_recursion_first_attempt.json`）**：用 amend3 的老数据 r5/r1c，结果 **v1 与 v2 逐位相同**（corr −0.4825/−0.3770/+0.2163）。原因：这两个 run 的抽样里**没有任何一步进入目标集**（当年到达率 ~0.05%），深井的生效条件（`g<0`）从未出现。**教训：门禁必须用含到达回合的数据。**
- **第二版（已跑完，结论 FAIL，但失败原因极其重要）**：改用 r12/r13 四个 run（到达回合 2.5%–7.1%），三条判据 S1 活跃性 / S2 方向 / S3 净改善。结果：

| run | 含目标集回合 | h_T（v1→v2） | corr(到达回合) v1 | corr(到达回合) v2 | S1/S2/S3 |
|---|---|---|---|---|---|
| r12_uncal_s0 | 26/900 | 0.0 → **−3.0** | −0.6588 | −0.6588（完全相同） | ✓/✗/✗ |
| r12_full_s0 | 40/900 | 0.325 → **−1.7** | +0.8260 | +0.8260（完全相同） | ✓/✓/✗ |
| r13_uncal_s0 | 24/900 | 0.0 → **−3.0** | −0.5030 | −0.5030（完全相同） | ✓/✗/✗ |
| r13_full_s0 | 56/900 | 0.411 → **−1.357** | +0.7932 | +0.7932（完全相同） | ✓/✓/✗ |

  `corr` 逐位相同不是巧合：**深井确实存在（h_T 变了），但没有进入策略优势**。已用探针 `scripts/t_probe_well_reach.py` 在三段真实到达回合上取证：

| 量 | v1 | v2 | 结论 |
|---|---|---|---|
| 步前 `h_t` 末三步 | 0.007 / 0.004 / 0.001 | 相同 | 到达步在目标集外（h 仍为正） |
| `h_T` | 0 / 0.325 | **−3 / −1.7** | 深井在终端确实存在 |
| `target_reach`（critic 通道） | 0.0 | **−3.0** | 深井**进了 reach 评价器** ✓ |
| `adv_total`（策略优势） | −0.0053 / −0.0027 / +0.0001 | **完全一致** | 深井**没有进策略梯度** ✗ |

  根因：联合 min 型递推在“停止点”取的是**步前状态**的 `h_t`（到达那一步的 t 侧距离仍 > r_goal，h 是正的小数），而“真正进入目标集的状态”只作为附加的 `h_T` 存在；到达步的 `done` 又截断了递推——于是**深井永远进不了策略优势**，只进了 `V_h` 的回归目标与可行性判据 `W`。参考实现没有这个问题，因为它的 `reach` 是**逐状态**函数，到达状态本身就是序列里的一行（post-step），停止价值自然等于 −300。

  修复候选已在探针里验证方向：把联合序列的 h 换成**步后状态**（`h_{t+1}`，终端补 `h_T`），到达步的优势立刻变负（三段样本：−0.056 / −0.145 / +0.0317，前两段显著转负）。这是**又一处契约级语义变更**（“停止价值取步后状态”），超出 r14 预登记范围，按纪律必须停下来请示。

## 5. 交接必须知道的关键事实

### 5.1 深井的终端数学（最容易被误读的一点）

联合即时信号 `g_τ = max(h_τ, −z_τ)`。到达时 `h_T = −3`，但终端值是 **`max(−3, −z_T) = −min(3, 剩余预算)`**。实测三组（合成探针）：

| 情形 | 终端目标值 |
|---|---|
| 干净到达、剩 0.4 预算 | **−0.4** |
| 差 1 m 停下 | **+0.0467** |
| 干净到达、剩 2.5 预算 | **−2.5** |

含义：**到达永远优于差一点停下；省下的电直接放大奖励**。这不是“被掩蔽”，是参考实现 `max(reach, −energy)` 的原生语义。所以深井的有效性必须用**含到达回合**的数据验证（第一版门禁的教训）。

### 5.1b 深井的作用范围（本轮实测，最重要）

| 通道 | 深井（h_T=−3）是否进入 | 证据 |
|---|---|---|
| `V_h` 回归目标（`target_reach`） | **是**（0 → −3.0） | 探针 + 门禁 |
| 可行性判据 `W`（干净到达 W=−3 ≤ −β） | **是** | HS5 单测 |
| 策略优势 `adv_total`（联合 min 递推） | **否**（v1≡v2 逐位相同） | 门禁 4 个 run + 探针 3 段 |
| 条件 | 深井只在“未违规的干净到达”上出现；带章到达 h_T=C·y（0.2–1） | 门禁 mean_h_T：r12_full 0.325→−1.7 说明部分回合有干净到达 |

**结论：若不做“停止价值取步后状态”的对齐，r14 的核心假设（深井恢复到达方向）在策略层不成立，开训大概率重演 r7–r13。**

### 5.2 h 语义 v2 与可行集

`safety_step_v2`：`y>0 → max(g, C·y)`；`y=0 → g`。可行集不变：违规步 `h ≥ C·floor = 0.2 > 0`（不可行），目标集外安全步 `h = g > 0`（未到达）。v1 与 v2 只在目标集内（`g<0`）有差异。

### 5.3 分级安全与撞死

非碰撞的越界（名义膨胀边界/墙带）按侵深给 severity ∈ [0.2, 1.0] 并做不可逆 max-历史；**真碰撞/出界仍是 failure 终止、h_T = FAIL_PENALTY = 1.0，语义未变**。`dead_step_mask` 阈值 0.5 不变：y≤0.5 的轻度违规步**重新获得梯度**（正是 F2 想要的），深违规步仍被屏蔽。

### 5.4 obs-v4 与旧 checkpoint

obs-v4 改了 z_ratio 编码边界，**旧 checkpoint 不能在新观测编码下直接评估**。r14 的评估脚本按 run 的预登记读环境参数；若要评估 r1–r13 旧权重，需用 `obs-v3` 编码（当前代码只支持按版本编码，需在评估入口显式指定）。

### 5.5 日志口径

- 训练进度/指标：`<run_dir>/train_log.jsonl`（**不是 .log**）。新行：`kind="episode_diag"`（每局）、`kind="update"` 内的 `win_diag`（每窗口）。
- 数值口径：`corr_g_adv > 0` 方向正确；`corr_c_adv` 在 w=0 时应 ≈0（成本只经 −z 进 max）。
- 训练器启动即打印 `[OBJ-FIX] … h_semantics=v2_deep_well`、`[STAGNATION] …`、`[SAFETY-CONT] enabled=True …`，**开跑前先核这三行**。

## 6. 未完成事项与下一步（按顺序）

**当前状态：r14 停在 T4 门禁，训练未启动。按预登记 (a) 条，门禁不过不得开训；需要一次决策。**

1. **决策点（需用户拍板）**：门禁失败暴露的不是“深井错了”，而是“深井只进了 critic、没进策略优势”。三个选项：
   - **A（推荐）**：对齐“停止价值取步后状态”——联合序列用 `h_{t+1}`（终端补 `h_T`）。参考实现本就是逐状态 post-step 语义；探针已证明方向转负。代价：一处契约级语义变更（需升版 + 预登记修订 + 门禁重跑），改动集中在 `storage.compute_objectives` 的联合序列构造，不动 critic 与 W。
   - **B**：照原预登记回退方案（终点超额一次性计费），不碰 min 递推——但这样策略优势仍拿不到到达信号，只是换了个成本罚的形式。
   - **C**：照现状开训——不建议，证据已表明大概率重演 r7–r13。
2. **门禁重跑**（若选 A）：改联合序列 → 更新 `test_h_semantics_v2`（HS6 改用步后口径）→ 重跑 `t_gate_r14_min_recursion_v2.py` 三条判据 → PASS 才进 smoke。
3. **T5 smoke**：6 个 run × 约 100 updates（`--smoke-updates`），检查：`win_diag.corr_g_adv` 不持续为负、无 NaN、`episode_diag` 与停滞现场正常、启动三行 `[OBJ-FIX]/[STAGNATION]/[SAFETY-CONT]` 正确。约 1 小时。
4. **T6 正式 r14**：6 run × 600 updates，`setsid` 启动（照 `scripts/run_round13.sh` 结构新建 `run_round14.sh`：2 方法 × 3 种子，GPU 3–7，每卡 ≤3）。预计 2–3 小时（load 40+ 时更久）。
5. **评估**：新建 `eval_round14.sh`（照 `eval_round13.sh`：`contract_margin_eval.py --t-max 45 --k 1.15`，run→prereg 指向 `training_loop_prereg_r14.json`），产出 `reports/training_round_r14.json`。
6. **T7 记录与提交**：工作记录追加到 plan_snapshot 并**立即回写服务器正本**（`/home/caohy/repositories/Paper/doc/plan/20261109_camra_navigation_implementation.json`，校验前 N 条逐字一致）；提交 `Paper` 仓库并 `git push origin main`。

## 7. 操作手册与已知的坑

**命令模板**

```bash
PY=/pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python
cd /home/caohy/repositories/Paper/CaMRA/navigation
export PYTHONPATH=src:../calibration/src
$PY tests/test_h_semantics_v2.py                  # 新测试
/tmp/run_all_tests.sh                            # 全量（PASS=32 FAIL=0）
$PY scripts/t_gate_r14_min_recursion_v2.py       # T4 门禁
```

**坑位清单（每条都踩过）**

1. 长任务必须 `setsid bash -c '… > log 2>&1' &`，否则 SSH 断连即死；
2. SSH 外层双引号会吃掉未转义的内嵌引号，导致补丁“静默不生效”——**补丁脚本写本地、断言替换成功再上传执行**；
3. `cmd | tail` 后 `$?` 是 tail 的退出码——要判测试成败必须直接跑并取 rc；
4. 服务器 load 常 40+（多人共用），并行别超过 8 个 run，且优先后台；
5. 训练目录规则：`run_dir` 已存在且非空直接拒绝，换 run id；
6. 上传的文件必须 `sed -i 's/\r$//'` 去 CRLF；
7. 同名进程可能残留（重复实例），启动前 `pgrep -f <脚本>` 清干净；
8. r13 的深坑：统计更新数用 `train_log.jsonl`，不要用 `.log` 行数。

## 8. 风险与回退

| 风险 | 应对 |
|---|---|
| 门禁 S2/S3 不过（已发生） | 已定位为“停止价值取步前状态”的语义缺口；选项 A 对齐步后状态（推荐）/ B 回退方案 A / C 硬训（不建议），见第 6 节 |
| 选项 A 改动引入新问题（联合序列整体平移一格） | HS6 单测改步后口径 + 门禁三条判据重验；不通过即回退 |
| 6 run 样本量不足以判定（种子间方差大） | 按预登记只看中位数；不足则如实报“无法判定”，不加轮次 |
| 服务器被同事占满 / SSH 不稳（实测 load 40+、连接偶发 SIGTERM） | 全部后台 `setsid` + 轮询；补丁脚本本地写、上传执行；不要交互跑长任务 |
| eval 口径与训练不一致（r12 的教训） | 评估参数一律从 run 的预登记读，已改在 `contract_margin_eval.py` |

## 9. 文件与提交状态

- 本地工作区：`C:\Users\smile\WorkBuddy\2026-11-09-21-29-26\nav064\r14\`（改动文件的工作副本、补丁脚本、本交接文档）
- 服务器仓库改动（**全部未提交**，按纪律在 T7 统一提交并推送）：
  - 代码 8 个：`observation.py`、`maps/geometry.py`、`envs/nominal.py`、`envs/vector.py`、`algorithm/targets.py`、`algorithm/storage.py`、`scripts/train_formal.py`、`scripts/contract_margin_eval.py`
  - 契约 2 个：`config/value_objective_contract_v1.json`（v1.3.0）、`config/navigation_contract_v1.json`（v1.6.0）
  - 预登记 2 个：`config/training_loop_prereg_r14.json` + `.sha256`
  - 测试：`tests/test_h_semantics_v2.py`（新），另 6 个测试文件被对齐/修红
  - 脚本：`scripts/t_build_r14_prereg.py`、`scripts/t_gate_r14_min_recursion.py`（首版）、`scripts/t_gate_r14_min_recursion_v2.py`、`scripts/t_probe_r12_arrivals.py`、`scripts/t_probe_well_reach.py`
  - 报告：`reports/gate_r14_min_recursion.json`（v2 门禁）、`reports/gate_r14_min_recursion_first_attempt.json`（首版证据）
- 已推送的基线：`main` = `36be38c`（计划正本回写 + 尺度审计脚本），远程一致
- 关键 sha：r14 预登记 `51a86ea3…`；值契约 v1.3.0 `c0032c2a…`；导航契约 v1.6.0 `332174c3…`

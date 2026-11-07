# HiRA 与 v2 统一口径对比（HIRA-037）

> **证据等级：报告级（未复算、非配对）**。v2 的逐 trial 原始数据已不在本机，
> 本报告不做任何 v2 复算；v2 数字全部来自其冻结产物与交接笔记，逐行标注出处。

## 1. 配置可比性

| 维度 | v2 | HiRA | 可比性 | 说明 |
|---|---|---|---|---|
| 命令域 | vx ±1.0 / vy ±0.4 / wz ±1.0 m/s,rad/s | 同（逐字段继承协议） | 可比 | 协议 v1.0.2 的命令域被 HiRA 协议逐字段继承 |
| 命令程序生成器 | camra_calibration.commands.build_program | 同一个函数（清单逐条按 program_sha256 重建对账） | 可比 | 同 seed + 同生成器 = 同命令表，这是跨控制器匹配的前提 |
| 五个 profile | clean/friction_only/mass_only/push_only/combined | 同五档，旋钮范围继承协议 | 可比 | profile 采样由 seed 确定性生成 |
| 机器人资产 | unitree_ros go2_description → USD | legged_gym go2.urdf → Isaac Lab 导入（29 links / 27 colliders） | **不可比** | 外形同为 Go2，但不是同一份资产；质量/惯量/碰撞体不同（§12、§14） |
| 执行器与 PD | kp=25.0, kd=0.5, action_scale=0.25 | kp=20.0, kd=0.5, action_scale=0.25（源 Go2 配置值） | **不可比** | kp 差 25%（沿用各自来源），这是与网络权重并列的第二个差异源 |
| 力矩限幅 | 见 v2 冻结包 deploy.yaml | 逐关节 URDF effort：hip/thigh 23.7、calf 35.55 N·m | **不可比** | 限幅不同会改变饱和行为 |
| 时序 | 物理 200 Hz / 策略 50 Hz / 命令 10 Hz | 同（且逐项实测：6000/1500/300/301） | 可比 | HiRA 侧用实测计数而非读配置（§18） |
| 能量口径 | applied_torque × 子步末速度，逐关节截断，float32 乘法 | 同公式，float64 乘法 | 可比（须注明） | 跨控制器比 J 时须注明 float32 vs float64（系统差 ~2.6e-10 相对，§25） |
| 协议版本 | calibration_protocol v1.0.2 | calibration_protocol_hira_v1（继承 v1.0.2，差异 9 项全为身份/簿记） | 可比（须注明） | 统计口径零变更；差异见 reports/protocol_diff_from_v2.json |
| 重试口径 | 含 D-007（baseline_validate +31 个 __a2）与 D-013（near +2）两次整批重采 | 1820 条全部单 attempt，零重采、零 resume | **不可比** | v2 的 heldout/near 分母是「按最大 attempt 去重后」的 397/197；HiRA 是原始 completed 377/188。**这是最重要的分母差异** |

## 2. 指标对比

| 指标 | v2（分母） | HiRA（分母） | 单位 | 可比性 |
|---|---|---|---|---|
| α（成本裕度） | 0.1905（790 completed） | 0.3516（771 completed） | ratio | 同口径 |
| δ（几何裕度） | 2.6139（同上） | 9.0322（同上） | m | 同口径 |
| heldout 成本覆盖 | 0.8690（345/397，按最大 attempt 去重） | 0.9098（343/377，completed 且可算） | 覆盖率 | **分母不同** |
| heldout 位置覆盖 | 0.9144（363/397，按最大 attempt 去重） | 0.8966（338/377，completed 且可算） | 覆盖率 | **分母不同** |
| heldout 联合覆盖 | 0.7935（315/397，按最大 attempt 去重） | 0.8170（308/377，completed 且可算） | 覆盖率 | **分母不同** |
| near 成本覆盖 | 0.8883（175/197，补采后按最大 attempt 计） | 0.9362（176/188，completed 且可算） | 覆盖率 | **分母不同** |
| near 位置覆盖 | 0.9340（184/197，补采后按最大 attempt 计） | 0.9840（185/188，completed 且可算） | 覆盖率 | **分母不同** |
| near 联合覆盖 | 0.8274（163/197，补采后按最大 attempt 计） | 0.9202（173/188，completed 且可算） | 覆盖率 | **分母不同** |
| near 几何重叠 | 138/197（197，含 2 条补采） | 151/200（200，completed+bf） | 条 / 百分比 | 同判据、**分母不同** |
| near 越界 | 13（197 可算） | 15（200） | 条 | 同判据、**分母不同** |
| 功率模型 validate q90 | 0.0554（n=79） | 0.0588（n=80） | 相对误差 | 同口径 |
| 足迹面积 / 基宽 | 0.4580（clean 轨迹重建） | 0.6769（623,757 帧 / 420 条） | m² / m | **资产不同** |
| 校准完整率（800 条） | 790/800（800） | 771/800（800） | completed/quota | 同口径 |
| 格级完整率缺口 | 20/20 ≥ 0.9（补采后）（—） | calibration 2 + heldout 3 + near 2（—） | 格数 | **重试政策不同** |

### 逐项注释与出处

- **α（成本裕度）**：同为 Q0.90(ε_c) nearest-rank；HiRA 是 v2 的 1.85×。不同控制器 + 不同资产 + 不同 kp，不能归因于单一变量
  - v2 来源：`CaMRA/calibration/docs/implementation_handoff.md §28 / config/margins_v1.json`
  - HiRA 来源：`config/margins_hira_v1.json（HIRA-031）`
- **δ（几何裕度）**：HiRA 是 v2 的 3.46×。膨胀主要来自扰动档（HiRA per-profile δ q90：clean 1.85 → combined 12.50）
  - v2 来源：`CaMRA/calibration/docs/implementation_handoff.md §28 / config/margins_v1.json`
  - HiRA 来源：`config/margins_hira_v1.json（HIRA-031）`
- **heldout 成本覆盖**：分母不同（397 vs 377）且 v2 含历史重采；只能作描述性比较，不能当同一批样本的配对差异
  - v2 来源：`CaMRA/calibration/docs/implementation_handoff.md §27 表（行 1819–1821）`
  - HiRA 来源：`reports/heldout_coverage.json（HIRA-033）`
- **heldout 位置覆盖**：分母不同（397 vs 377）且 v2 含历史重采；只能作描述性比较，不能当同一批样本的配对差异
  - v2 来源：`CaMRA/calibration/docs/implementation_handoff.md §27 表（行 1819–1821）`
  - HiRA 来源：`reports/heldout_coverage.json（HIRA-033）`
- **heldout 联合覆盖**：分母不同（397 vs 377）且 v2 含历史重采；只能作描述性比较，不能当同一批样本的配对差异
  - v2 来源：`CaMRA/calibration/docs/implementation_handoff.md §27 表（行 1819–1821）`
  - HiRA 来源：`reports/heldout_coverage.json（HIRA-033）`
- **near 成本覆盖**：同上：分母 197 vs 188，且 v2 含 2 条补采
  - v2 来源：`CaMRA/calibration/docs/implementation_handoff.md §32.4（行 2063–2065）`
  - HiRA 来源：`reports/near_obstacle_evaluation.json（HIRA-035）`
- **near 位置覆盖**：同上：分母 197 vs 188，且 v2 含 2 条补采
  - v2 来源：`CaMRA/calibration/docs/implementation_handoff.md §32.4（行 2063–2065）`
  - HiRA 来源：`reports/near_obstacle_evaluation.json（HIRA-035）`
- **near 联合覆盖**：同上：分母 197 vs 188，且 v2 含 2 条补采
  - v2 来源：`CaMRA/calibration/docs/implementation_handoff.md §32.4（行 2063–2065）`
  - HiRA 来源：`reports/near_obstacle_evaluation.json（HIRA-035）`
- **near 几何重叠**：同一判据（足迹与障碍几何重叠）：v2 70.1%，HiRA 75.5%，同一量级
  - v2 来源：`CaMRA/calibration/docs/implementation_handoff.md §32.4 与 §31.1`
  - HiRA 来源：`reports/near_obstacle_evaluation.json（HIRA-035）`
- **near 越界**：场地边界墙两边都未建模，均为几何余量判定
  - v2 来源：`CaMRA/calibration/docs/implementation_handoff.md §32.4`
  - HiRA 来源：`reports/near_obstacle_evaluation.json`
- **功率模型 validate q90**：各自独立拟合、独立验证；两套模型不可混用，J_nom 不能跨控制器互换
  - v2 来源：`CaMRA/calibration/docs/implementation_handoff.md 行 1259`
  - HiRA 来源：`config/power_model_hira_v1.json（HIRA-026，一次性 validate）`
- **足迹面积 / 基宽**：HiRA 面积为 v2 的 1.48×、基宽 1.32×；资产不同，轮廓不可互用
  - v2 来源：`CaMRA/calibration/config/footprint_v1.json`
  - HiRA 来源：`config/footprint_hira_v1.json（HIRA-027）`
- **校准完整率（800 条）**：v2 有 10 条 terminated、HiRA 有 29 条业务失败（其中 160 条噪声事故已零删除隔离后重采）
  - v2 来源：`CaMRA/calibration/docs/implementation_handoff.md 行 1757`
  - HiRA 来源：`reports/calibration_collection_integrity.json`
- **格级完整率缺口**：v2 达到 20/20 靠的是 D-013 一次性补采（历史特例，§9.3）；HiRA 按新契约不补采，缺口如实留存。**这不是能力差异，是重试政策差异**，不能直接比
  - v2 来源：`CaMRA/calibration/docs/implementation_handoff.md §32.4（cells_below_share = {}）`
  - HiRA 来源：`reports/{calibration_collection_integrity,heldout_collection_integrity,near_obstacle_collection_integrity}.json`

## 3. 结论（描述性，不作因果）

- α 0.1905 → 0.3516（1.85×）、δ 2.6139 → 9.0322 m（3.46×）：HiRA 在 30 s 开环、受扰动时漂得更远。
- heldout：成本 86.90% → 90.98%（+4.08 pp），位置 91.44% → 89.66%（−1.78 pp），联合 79.35% → 81.70%（+2.35 pp）。成本变好、位置变差，方向不一致，**不能概括为「更好」或「更差」**。
- near：成本 88.83% → 93.62%、位置 93.40% → 98.40%、联合 82.74% → 92.02%，但这两组 δ 差 3.46 倍，覆盖率被更大的 δ 抬高了——**δ 越大越容易「覆盖」，这不是性能改善**。
- near 几何：重叠 70.1% → 75.5%、越界 13 → 15，同一量级。两边都说明 δ 是必要条件而非充分条件。
- 格级缺口：v2 20/20（靠一次性补采）vs HiRA 17/20（不补采）。这是重试政策差异，不是能力差异。

## 4. 边界与不可比项

- **报告级，未复算**：v2 的逐 trial 原始数据已不在本机（/pub 只剩 3.1 GB TASK 日志），本报告不做任何 v2 复算，全部 v2 数字来自其冻结产物与交接笔记（逐行标注出处）。
- **非配对**：只能比汇总数字，做不了「同 seed 同命令」的成对差异，也无法给簇级区间。清单层面 baseline/heldout/near 与 v2 有 100% 模板匹配（§26），但缺 v2 的逐 trial 轨迹与扰动 trace，配对检验无从做起。
- **多重差异源**：控制器包同时更换了权重、默认姿态、执行器增益（kp 25→20）与机器人资产。任何指标差异都不能归因于网络权重一项。
- **分母口径**：v2 的 397 / 197 是「按最大 attempt 去重后」，含历史重采；HiRA 的 377 / 188 是原始 completed。引用时必须带分母。
- **δ 不可横比覆盖**：δ 从 2.61 涨到 9.03 会自动抬高 near 的「位置覆盖」，把 93.40% → 98.40% 当性能提升是错的（§37 已证明 98.40% 覆盖下仍有 75.5% 几何重叠）。
- **旧报告未改**：v2 的交接笔记、计划、锁、报告一字未动；本报告只读引用。

## 5. 一处更正

更正 HIRA-033 的 remaining_work 引用错误：该处写「v2 的 CAL-027（cost 0.8883 / position 0.9340 / joint 0.8274，n=197）」有误。0.8883/0.9340/0.8274 是 v2 **近障**（§32.4，n=197 补采后）的数字；v2 的 CAL-027（heldout）实为 0.8690 / 0.9144 / 0.7935（n=397，§27）。本报告按后者引用。


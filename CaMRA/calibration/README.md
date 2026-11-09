# CaMRA 校准目录（camra_calibration）

本目录是 CaMRA（四足机器人成本/几何裕度校准）的**证据与工具目录**，不包含稿件、
不改动 `unitree_rl_lab` 与冻结控制器本体。

## 当前状态：后续固定使用 v2 配套校准

用户于2026-11-09确认：后续开发固定使用 **`camra_go2_velocity_it2999_v2`**，作为硬性约束。主线校准复核和导航消费必须绑定其独立的功率模型、足迹、α/δ、协议和数据锁；入口为本目录 `config/power_model_v1.json`、`footprint_v1.json`、`margins_v1.json`、`data_lock_v1.json`，使用前核验锁与对应代码版本。HiRA 实验保留为历史参考/对照，不能将其校准值或原始执行数据用于 v2；控制器切换须经用户明确授权，完整约束见[根 AGENTS](../../AGENTS.md)。

当前入口为[60项上层导航计划](../../doc/plan/20261109_camra_navigation_implementation.json)，版本 **1.2.0**，**0/60项完成**，从 **NAV-001** 开始。该阶段消费冻结 v2 的配套结果，先核验速度域、低速执行和闭环裕度适用性，不默认重采校准；执行节点为 tiger11（192.168.20.11），服务器任务使用 tmux。

历史 [40项 HiRA 接入与校准计划](../../doc/plan/20261105_camra_hira_policy_calibration.json)已记录 **40/40 项完成**。G1 接入门禁 16/16、独立审计 12/12 和独立目录重载 20/20 均有报告；pilot、400 条功率基线、800 条校准、400 条检验与 200 条近障配额已采完，α/δ 已冻结，最终交接已完成。任务完成不表示全部科学指标达标。

现有v2校准工具及结果已经建立，原32项计划记录完成。HiRA 是独立候选，适配已完成并冻结（当前检验锁 v3 `d7d60cb9`；heldout 400 条归属 v1 `289e4137`，near 200 条归属 v2 `db156f4d`，历史归属不改）；`experiments/hira_go2_18500/` 保存新配置、七清单、冻结包、报告与交付索引，适配实现位于相邻 locomotion 模块。HiRA 的采集入口是 `tests/run_delivery.py` 或 `tests/run_split.py` + 对应 stage lock（**两者都带输出目录绑定守卫**：目标 run 目录已绑别的 split/锁/包/清单时 exit 2 拒绝），**不能沿用本页 v2 命令**。

[控制器选择与复核记录](docs/controller_comparison_status_20261109.md)保存历史边界：HiRA 的批间关节/上一动作复位、warmup 后名义初态对齐、全体分母和交付索引仍待复核，尚未实施修复或重校准。这些历史 HiRA 待办不默认阻塞 v2 开发；共享工具的问题须检查对 v2 的实际影响。迁移报告只证明 HiRA 数据在 tiger11 校验通过，不代替 v2 资源核验。

历史 HiRA 阶段采用30秒测量窗口和正关节机械功（J）定义，独立完成20条pilot、400条clean功率基线（240/80/80）、800条校准、400条检验及200条近障检查。先验证并冻结功率模型和正式数据锁，再校准α/δ，冻结裕度后才采集检验与近障结果；该顺序已记录完成，1820条配额不作为当前导航的新任务。

新控制器、协议、功率模型、足迹、裕度、数据锁、trial ID及输出目录分别绑定同一实验身份。旧v2数值只作历史对照，不能作为HiRA结果；本轮论文资料始终只读。

## 一条链路在做什么

```
协议 → 清单 → 采集 → 功率模型 → 边际(alpha, delta) → 检验覆盖 → 近障评估 → 审计 → 交付包
```

下表列出现有 **v2** 产物；HiRA 已生成独立版本，不能沿用这些文件作为 HiRA 的冻结结果。

| 阶段 | 关键产物 | 说明 |
|---|---|---|
| 协议 | `config/calibration_protocol_v1.0.2.json` | 时钟、命令域、profile、配额、统计口径、准入规则 |
| 清单 | `doc/plan` + `/pub/.../trial_lists_v1.0.2/*.jsonl` | 七分片、1820 条，seed 块两两互斥 |
| 采集 | `scripts/collect_trials.py` + `sim/` | 冻结 v2 策略 + 逐 env 终止语义 + 可选障碍注入 |
| 功率模型 | `config/power_model_v1.json` | 只看速度命令预测能耗（NNLS，validate q90 5.5%） |
| 足迹 | `config/footprint_v1.json` | 冻结资产 + 60 万帧凸包，0.889×0.585 m |
| 边际 | `config/margins_v1.json` | alpha=0.19048、delta=2.6139 m（nearest-rank + bootstrap） |
| 数据锁 | `config/data_lock_v1.json` | 协议/模型/足迹/接口/margins/清单/地图/代码树全绑定 |
| 评估 | `scripts/evaluate_{heldout,near_obstacle}.py` | 成本/位置/联合覆盖、四套分母、按格与子组 |
| 审计 | `scripts/audit_all_splits.py` | 独立 numpy 重算能量 + 配额/seed/散列 |
| 交付 | `delivery/bundle_manifest_v1.json` | 路径 + 散列索引，仓库外大文件只引用 |

HiRA 在同一链路下的独立产物（均在 `experiments/hira_go2_18500/` 内，**不与上表混用**）：

| 阶段 | 关键产物 | 结论 |
|---|---|---|
| 协议 | `config/calibration_protocol_hira_v1.json` | 时钟、命令域、五 profile、配额、统计口径 |
| 清单 | `manifests/*.jsonl` + `lock.json` | 七分片 1820 条，seed 块两两互斥 |
| 采集 | `tests/run_delivery.py` → `run_split.py` | 1820 条全部单 attempt（零重采），失败样本零删除 |
| 功率模型 | `config/power_model_hira_v1.json` | 十参数，validate q90 5.88%（n=80） |
| 足迹 | `config/footprint_hira_v1.json` | 52 顶点、0.6769 m²、最小宽度 0.7596 m |
| 边际 | `config/margins_hira_v1.json` | **α 0.351568、δ 9.032231 m**（n=771，nearest-rank + 簇 bootstrap） |
| 数据锁 | `config/data_lock_hira_{calibration_v2,evaluation_v2,evaluation_v3}.json` | 采集 `c229c417`；检验 v2 `db156f4d`（near 200 条归属）、v3 `d7d60cb9`（当前，含 run_split 守卫）；heldout 400 条仍归 v1 `289e4137`；margins 全部指向同一个冻结文件 |
| 评估 | `tests/coverage_heldout.py`、`tests/evaluate_near_obstacle.py` | 完整且可算条件覆盖：heldout n=377 为 0.9098/0.8966/0.8170；近障 n=188 为 0.9362/0.9840/0.9202；几何重叠 151/200、δ 条件 **0/200** |
| 审计 | `tests/full_independent_audit.py` | 12/12，能量 1820 条最差 rel 4.97e-16，α/δ 逐位一致 |
| 对照 | `tests/compare_hira_v2.py` | 与 v2 **报告级**对比（v2 逐 trial 数据已不在本机，recomputed=false） |
| 交付 | `delivery/bundle_manifest_hira_v1.json` + `docs/reproduction.md` | 历史索引 231 文件 / 1880 trial；独立目录重载 20/20；当前 Git 字节复核例外及后续文档散列差异见最新复核说明 |
| 稿件 | `reports/manuscript_sync_differences.md` | 只读差异清单，未改稿件 |

## 现有 v2 工具命令

以下用于理解和复核历史 v2 链路，执行前按实际设备配置路径并进入tmux；不是当前导航的启动命令，也不是 HiRA 入口。当前不默认重新采集或生成冻结产物，导航按 NAV 计划建立隔离适配与输出。

```bash
# 采集（正式 split 必须带锁；combined 需 --bucket on 单独一趟）
python scripts/collect_trials.py --manifest-dir $MAN --split calibration --bucket off \
  --out-root $OUT --num_envs 32 --lock config/data_lock_v1.json

# 覆盖率 / 审计 / 交付包
python scripts/evaluate_heldout.py --manifest-dir $MAN --out-root $OUT --report $R1
python scripts/evaluate_near_obstacle.py --manifest-dir $MAN --out-root $OUT --report $R2
python scripts/audit_all_splits.py --manifest-dir $MAN --out-root $OUT --lock config/data_lock_v1.json --report $R3
python scripts/build_delivery_bundle.py --manifest-dir $MAN --logs-root $LOGS --out delivery/bundle_manifest_v1.json
python scripts/verify_delivery_bundle.py --manifest delivery/bundle_manifest_v1.json --report $R4

# 测试
python -m pytest tests -q
```

## 三条硬规矩

1. **不改冻结值**：协议、功率模型、足迹、margins 一旦冻结，后续只能用不能改；
   检验/近障的结果不得回填 alpha、delta 或模型（协议 `heldout.may_not`）。
2. **锁与代码一致**：`scripts/build_data_lock.py` 绑定代码树散列。代码改变后为新实验生成新版本锁，不能覆盖旧锁；历史复算使用原锁对应的只读源码快照。
3. **失败与原始记录保留**：旧数据、作废数据及全部attempt不删除、不覆盖。新实验预登记主attempt和恢复规则，业务失败不以最好或最高attempt替换；旧v2一次性补采特例不推广。

## 原 v2 的已知限制与结果版本

* 冻结 v2 速度跟踪器**没有避障能力**，近障补采后可计算记录中138/197条几何重叠；
  δ 是危险集扩张的必要条件而非充分条件，可用性取决于尚未实现的上层导航策略。
* `straight_shift|clean` 原完整率0.80；用户一次性裁定补采后为0.90，20格均达到旧协议完整率门槛。原失败与重试均保留，不能据此给新实验默认补到成功。
* 场地边界墙未建模，越界用几何余量检查代替；§32更新结果为13条越界，补采前报告仍保留。

详细版本和分母见 [旧校准交接笔记](docs/implementation_handoff.md)，补采及更新结果见§32。HiRA 已有独立运行报告，但两者控制器包、裕度、重试政策与分母不同，现有报告不能支持全面优劣结论。HiRA 按全体 400 条独立检验计，成本/位置/联合覆盖为 85.75%/84.50%/77.00%；该值由保存计数推得，未重算原始轨迹及区间，详见最新复核说明。

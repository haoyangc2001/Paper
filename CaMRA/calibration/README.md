# CaMRA 校准目录（camra_calibration）

本目录是 CaMRA（四足机器人成本/几何裕度校准）的**证据与工具目录**，不包含稿件、
不改动 `unitree_rl_lab` 与冻结控制器本体。

## 当前任务：HiRA 来源策略的独立校准

当前入口为 [40项 HiRA 接入与校准计划](../../doc/plan/20261105_camra_hira_policy_calibration.json)，**38/40 项完成**（HIRA-001–038），剩 HIRA-039 与 HIRA-040。G1 接入门禁 16/16 已通过；pilot、400 条功率基线、800 条校准、400 条检验与 200 条近障均已采完并留有完整性报告；α/δ 已冻结，交付包已建并通过独立目录重载 20/20。

现有v2校准工具及结果已经建立，原32项计划记录完成。HiRA 是新增独立候选，适配已完成并冻结（检验锁 `db156f4d`，heldout 400 条仍归属 v1 `289e4137`）；`experiments/hira_go2_18500/` 已建立，保存新配置、七清单、冻结包、报告与交付索引，适配实现位于相邻 locomotion 模块。HiRA 的采集入口是 `tests/run_delivery.py`（带输出目录绑定守卫）或 `tests/run_split.py` + 对应 stage lock，**不能沿用本页 v2 命令**。

本轮保持30秒测量窗口和正关节机械功（J）定义，独立完成20条pilot、400条clean功率基线（240/80/80）、800条校准、400条检验及200条近障检查。先验证并冻结功率模型和正式数据锁，再校准α/δ，冻结裕度后才采集检验与近障结果——以上已按此顺序完成。执行主机终端任务使用tmux。

新控制器、协议、功率模型、足迹、裕度、数据锁、trial ID及输出目录分别绑定同一实验身份。旧v2数值只作历史对照，不能作为HiRA结果；本轮论文资料始终只读。

## 一条链路在做什么

```
协议 → 清单 → 采集 → 功率模型 → 边际(alpha, delta) → 检验覆盖 → 近障评估 → 审计 → 交付包
```

下表列出现有 **v2** 产物；HiRA将生成独立版本，不能沿用这些文件作为新候选的冻结结果。

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
| 数据锁 | `config/data_lock_hira_{calibration_v2,evaluation_v2}.json` | `c229c417` / `db156f4d`，margins 冻结、代码 38 文件 |
| 评估 | `tests/coverage_heldout.py`、`tests/evaluate_near_obstacle.py` | heldout 0.9098/0.8966/0.8170；近障 0.9362/0.9840/0.9202、几何重叠 151/200、δ 条件 **0/200** |
| 审计 | `tests/full_independent_audit.py` | 12/12，能量 1820 条最差 rel 4.97e-16，α/δ 逐位一致 |
| 对照 | `tests/compare_hira_v2.py` | 与 v2 **报告级**对比（v2 逐 trial 数据已不在本机，recomputed=false） |
| 交付 | `delivery/bundle_manifest_hira_v1.json` + `docs/reproduction.md` | 220 文件 / 1880 trial 索引；独立目录重载 20/20 |
| 稿件 | `reports/manuscript_sync_differences.md` | 只读差异清单，未改稿件 |

## 现有 v2 工具命令

以下用于理解和复核旧链路，执行前按实际设备配置路径并进入tmux。它们不是HiRA现成入口；新入口须先完成计划规定的显式配置、拒混和数据锁验证。

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

详细版本和分母见 [旧校准交接笔记](docs/implementation_handoff.md)，补采及更新结果见§32。本轮HiRA是否改善这些指标尚无运行证据。

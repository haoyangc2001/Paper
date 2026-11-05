# CaMRA 校准目录（camra_calibration）

本目录是 CaMRA（四足机器人成本/几何裕度校准）的**证据与工具目录**，不包含稿件、
不改动 `unitree_rl_lab` 与冻结控制器本体。

## 当前任务：HiRA 来源策略的独立校准

当前入口为 [40项 HiRA 接入与校准计划](../../doc/plan/20261105_camra_hira_policy_calibration.json)，0/40项完成，首项HIRA-001。先学习现有实现和HiRA源码，在Isaac Lab完成模型、观测、关节映射、PD、时序、闭环及能量计量验证；HIRA-020接入门禁通过前不开展pilot、功率基线或正式采集。

现有v2校准工具及结果已经建立，原32项计划记录完成。HiRA是新增独立候选，尚未完成适配；现有采集器仍有v2模型路径和默认协议等绑定，不能只替换权重就按本页旧命令采集。计划中的 `experiments/hira_go2_18500/` 尚未建立，后续在该处保存新配置、报告和交接，适配实现位于相邻locomotion模块。

本轮保持30秒测量窗口和正关节机械功（J）定义，独立完成20条pilot、400条clean功率基线（240/80/80）、800条校准、400条检验及200条近障检查。先验证并冻结功率模型和正式数据锁，再校准α/δ，冻结裕度后才采集检验与近障结果。执行主机终端任务使用tmux。

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

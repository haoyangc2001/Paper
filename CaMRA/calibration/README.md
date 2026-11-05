# CaMRA 校准目录（camra_calibration）

本目录是 CaMRA（四足机器人成本/几何裕度校准）的**证据与工具目录**，不包含稿件、
不改动 `unitree_rl_lab` 与冻结控制器本体。

## 一条链路在做什么

```
协议 → 清单 → 采集 → 功率模型 → 边际(alpha, delta) → 检验覆盖 → 近障评估 → 审计 → 交付包
```

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

## 常用命令

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
2. **代码改动要重建锁**：`scripts/build_data_lock.py` 绑定代码树散列，改了任何 `.py`
   都必须重建，否则正式采集会被门禁 REFUSED。
3. **零删除**：旧的、作废的、失败的数据一律改名封存（如
   `trials/calibration_superseded_batchabort`），不删。

## 已知限制（如实记录）

* 冻结 v2 速度跟踪器**没有避障能力**，近障 5–35 cm 净距下 138/196 条几何重叠；
  δ 是危险集扩张的必要条件而非充分条件，可用性取决于尚未实现的上层导航策略。
* `straight_shift|clean` 完整率 0.80 < 0.9，按协议该 cell 不得进入正式结论。
* 场地边界墙未建模，越界用几何余量检查代替（14 条越界已如实上报）。

详细过程见 `docs/implementation_handoff.md`（§1–§34）。

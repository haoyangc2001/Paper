# NAV-064 r14：候选修复复核与重训提案

## 最新启动证据：r14a2 正在训练

2026-11-15 23:47:33，六个新 run 已从本地提交 `4b793d1d59e97d3f1c7db9f7dfc2252279852109` 在 tiger11/tmux 的 `nav064-r14-repair-20261115:r14a2-training` 启动。每组已完成至少两个正式更新；独立检查六组前两次更新的共49032个原始样本，确认18×227窗口、10 epoch×512、gamma_h由0.999升至0.9990033、gamma_joint=1及原始回合身份/写入hash匹配。具体PID、进度与路径见 `reports/nav064_r14a2_started_v1.json`。

协调器会在首100更新后严格审计，在各600更新结束后全量审计并评估最后checkpoint的development表现。当前训练与性能验收均未完成，NAV-064仍false，NAV-065及后续未启动。旧attempt与失败全保留。下方“等待提交后开训”是本次启动前的历史放行状态。


## 最新执行状态：r14a2 已放行，等待本地提交后开训

用户已明确批准一次新增六个 fresh-init run ×600 更新及必要本地提交。旧六个有实现偏差的 attempt 已按 PID 核验后结束，368 个完整更新、未完成窗口、原日志、原始数组和 u50 checkpoint 均保留；旧报告和冻结预登记不改写。保存与终止证据为 `reports/nav064_r14a2_preparation.json`。

修复已接入正式源码：reach 使用参考实际 gamma_2 双倍调度，半程达到 0.99999；reset 前保存回合标量身份；训练、评估、审计从冻结 registry 读取本次 run ID。新门禁 65/65、32 套回归全部通过，放行报告为 `reports/nav064_r14a2_training_release_v1.json`。活动源码、消费链和新门禁已绑定 `config/training_loop_prereg_r14_repair_v2.json`（r14.2.0，SHA256 `87e814ae0de8aee5276627232e6eff4c8d77b339a9e8568e16e7e337aa19eff2`），放行快照见报告。

新 run 为 `nav064_r14a2_{uncalibrated,full}_s{0,1,2}`，每 run 600 更新、每更新18×227=4086样本；前100计入600。入口 `scripts/run_round14a2.py` 在 tiger11/tmux 核验预登记与重名后启动六组，自动执行第100更新前缀审计、600更新全量审计及最后 checkpoint 的 development 评估。失败会保留现场并停下；不会自动推进 NAV-065 或追加新轮次。输出根为 `/pub/data/caohy/work/camra_navigation/`，本轮协调日志目录为 `nav064_r14a2/`。

目前只通过训练前验收，NAV-064 仍 false。两方法三种子中位数可行 SR≥0.10、能耗/预算≤1、full危险步中位数为0及机制/稳定性/数据审计需全部满足才可关闭。corr 只作诊断，不能替代门禁或性能。固定v2、旧校准、预算选择集与final封存、只读参考和论文边界保持。

下列“暂停”“待授权”“尚未替换”等文字均保留为此前时点的历史记录，当前资源授权已解除。


归属主计划 1.3.1 的 NAV-064。本文记录具体候选与资源决定，不增加 NAV 任务，也不另建完成状态。真实工作记录继续追加到主计划当前任务。用户允许在当前 tiger11 会话直接用 tmux 执行的决定继续有效；当前待决仅是实现缺陷后的新增重训 attempt 与额度。

## 已确认的问题和影响

正式入口始终使用 `gamma_h_init=0.999`，未消费已冻结的 `gamma_h_final=0.99999`。参考 `RC-PPO/rl/EC-EFPPO.py` 实际消费的是 `gamma_2 → xs[:,1] → ent_gamma[1]`，半程达到 final。旧 37 项门禁与 32 套回归均没有覆盖这一调度接线；它们的历史 PASS 保留，不能作为完整生产链已正确的依据。

真实 update50 的固定折扣可重现存储 reach 目标；示意的一倍线性调度在 full_s0 上产生最大约 0.00095636 的目标差。该探针只证明运行值会影响目标，不是参考双倍调度的数值对照。候选和专项测试采用已追踪的双倍公式。

另一个问题发生在目标已经计算后：`window_segments` 持有可变 `EpisodeState`，reset 后写摘要时借用了下一回合 episode_id 和预算。原始逐步数组及 episode_diag 正确。前 5 更新六 run 共 122580 样本的独立审计为 `PASS_WITH_RECONSTRUCTED_METADATA`，每 run 的 90 处摘要错配均在独立索引中重建；原 manifest 不改写。

## 现场与候选

2026-11-15 23:03:23 已逐 PID 验证身份后 SIGSTOP 六个进程，当前已完成更新数为：

| 方法 | seed0 | seed1 | seed2 |
| --- | --- | --- | --- |
| uncalibrated | 65 | 62 | 64 |
| full | 59 | 59 | 59 |

共 368 个完整更新，另有暂停时尚未完成的窗口。PID 为 2550623–2550628；会话 `nav064-r14-repair-20261115`。原日志、原始数组、六个 `state_u50.pt` 和内存现场均保留。原预登记 r14.1.0 SHA256 为 `30033056d6ca2b27d636998ee9d85b573e791e53ceee4a78c3e127299a206c0a`。活动源码与该冻结文件尚未替换；不直接 SIGCONT 这批进程。

候选目录为 `/pub/data/caohy/work/camra_navigation/nav064_r14_candidate_gamma_meta_v2/CaMRA/navigation`，仅 `scripts/train_formal.py` 的生产实现做两项修正：

1. 每次 update 在课程配置重绑后设置 reach 折扣：`min(final, init + (final-init) * 2*update_idx/total_updates)`。合同只在循环外读取一次；`gamma_cost=gamma_joint=1` 保留。
2. 在 reset 前捕获 episode_id、map、k、z0 的标量快照；写 chunk 时验证原始 episode_id，并记录 update_idx、gamma_h 和 gamma_joint。更新日志也记录两个折扣。

候选源码 SHA256：`f993fcf9bf6836cd9ef25662bcf7fb1731dfa0b6482e2e2a04574f2689dd8ada`。当前可应用补丁为 [train_formal_r14_gamma_meta_v3.patch](patches/train_formal_r14_gamma_meta_v3.patch)，已用 `git apply --check` 确认。早期 v2 补丁保留但不是最终候选，不能用它代替 v3。

## 已执行的复核

专项 14 项验证实际循环中的调度赋值、半程钳位、reach 目标消费、joint/cost 独立性以及 reset 后实际写循环的旧回合身份和预算。独立数值门禁 37/37 PASS；候选源码绑定匹配。针对性回归 8/8 PASS：training_semantics、rollout_resume、budget_selection、value_objective、minimization_update、targets_oracle、network_contract、observation_contract。

历史错误保留：第一次回归漏带已暴露的历史 v1 delivery/report fixture，budget_selection 因缺文件退出；另一个套件名误写成 minimization_ppo，退出 2。补齐隔离目录中的旧 fixture 并用正确名称复验通过；没有执行新预算选择。复现脚本 v3 的 14 项检查通过后，输出时因 CLI 参数变量被测试循环覆盖而报错；v4 单独修正该核验器变量名，生产候选没有变化，v3 脚本与日志保留。

可复现入口为 [verify_train_formal_r14_gamma_meta_v4.py](patches/verify_train_formal_r14_gamma_meta_v4.py)。须在同一 tiger11 tmux 中用绝对解释器执行，并指定未存在的输出文件：

```bash
/pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python \
  /home/caohy/repositories/Paper/CaMRA/navigation/docs/patches/verify_train_formal_r14_gamma_meta_v4.py \
  --navigation-root /pub/data/caohy/work/camra_navigation/nav064_r14_candidate_gamma_meta_v2/CaMRA/navigation \
  --source-sha256 f993fcf9bf6836cd9ef25662bcf7fb1731dfa0b6482e2e2a04574f2689dd8ada \
  --out <新的专项报告绝对路径>
```

当前报告为 `reports/nav064_r14_gamma_meta_candidate_v4.json`；候选 37 项报告为其目录内 `reports/gate_gamma_meta_v2.json`。回归日志在候选根的 `regressions/`，初次退出码与 a2 退出码分别保存。汇总绑定报告为 `reports/nav064_r14_gamma_meta_review_v1.json`。这些证据只说明候选修复的数值和回归正确性，不能证明训练、开发性能或 NAV-064 关闭档已通过。

## 待确认的具体重训提案

建议保留原六个中断 attempt 为实现偏差诊断；批准后记录归档与终止原因，结束暂停进程及其等待器。以新预登记 `training_loop_prereg_r14_repair_v2.json`（拟建 r14.2.0）绑定最终源码、包含调度消费的新门禁和匹配评估/审计入口，随后开展六个 fresh-init run。拟用 ID 如下，启动前逐一拒绝重名：

| 方法 | seed | 新 run ID |
| --- | --- | --- |
| uncalibrated | 0 | nav064_r14a2_uncalibrated_s0 |
| uncalibrated | 1 | nav064_r14a2_uncalibrated_s1 |
| uncalibrated | 2 | nav064_r14a2_uncalibrated_s2 |
| full | 0 | nav064_r14a2_full_s0 |
| full | 1 | nav064_r14a2_full_s1 |
| full | 2 | nav064_r14a2_full_s2 |

每 run 600 更新、每更新 18×227=4086 样本，合计另 3600 个完整更新、14709600 样本；前 100 更新计入 600，不另跑性能 smoke。原 attempt 的已完成 368 更新及未完成窗口额外记账，累计正式/中断 attempt 变为 12 个，不能写成总共只跑 6 个。这是实现缺陷后重训，不是已授权的基础设施重跑，也不以取消墙钟上限解释为无限增加 run。

保持 fresh init、最后 checkpoint、保存周期 50、原两方法/三种子、train/development、地图、预算抽样、T45、原开发关闭档及固定 v2 全套不变。拟修订 evaluator 显式消费新 run ID；全量审计按每个 chunk 的 update 和已冻结双倍调度重算 gamma_h，并严格验证新摘要不再错配，不能直接复用旧审计器的固定 0.999 假设。新门禁/预登记须覆盖这些实际消费者的源码绑定，全部一致后才可启动。

最终仍要求两方法分别三种子中位数安全可行 SR≥0.10、能量/预算中位数≤1，full 危险步中位数为 0，并通过机制、稳定性、原始样本和数据访问审计。预算选择与 final_* 不用于训练验收；未满足时 NAV-064 保持 false，不推进 NAV-065。不自动启动 r15 或额外候选。

需要用户确认的依据：主计划 `uncertainty_policy` 明确将“改变正式试验次数”列为暂停点；`upstream_correction_policy` 允许普通实现修复但不允许自行扩充资源。根 AGENTS 的开工要求同样规定涉及资源的实质未决变化先询问。当前建议批准上述一次新增六 run 完整重训；另一选择是继续保持暂停并由用户裁定额度。未经确认不将推荐视为批准。

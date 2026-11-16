# NAV-064 安全净空接口回修（2026-11-16）

本次任务仍是计划1.3.1的NAV-064，依赖NAV-061–063；目标是在固定v2底层上训练预算内安全到达的上层导航器。当前只完成接口回修及数值/回归验收，NAV-064保持false，NAV-065及后续不放行。

## 上一轮已结案，仍未达标

time-feedback-v1六组各600更新已全部完成，总计3600更新、14709600样本；600更新全量审计PASS，开发结论FAIL_DEVELOPMENT。主预算k=1.15下：

| 方法 | 可行成功率三种子中位数 | 能耗/预算中位数 | 危险步三种子中位数 | 性能判定 |
| --- | --- | --- | --- | --- |
| uncalibrated | 1/6 | 0.863326 | 0 | 通过本方法指标 |
| full | 1/2 | 0.890408 | 228 | 未通过安全指标 |

两方法必须同时通过，不能按到达率单独报成功。full的18个主预算回合中，8个安全可行到达、7个超时、2个碰撞，另1个到达但曾轻度越界；仅1个回合超预算。当前问题主要是安全路径和按时到达，不能再仅归因为预算太紧。

`scripts/diagnose_clearance_feedback.py`只读复算最新train的预定更新99/349/599及已有development报告：full共36774条样本，首次越界60次，其中54次即时max仍由距离项占优；不可逆历史饱和且非failure的2726条样本全部stop-now、联合target=1。这是局部信号及历史平台的证据，不表示首次越界完全没有反馈：现有延续值下界仍能影响越界动作的优势。不能据此清空安全历史、屏蔽危险样本或直接调大奖惩权重。

## 单一候选与生产接口

obs-v7在obs-v6原41个字段之外增加20个字段：16个按机身方位归类的障碍安全净空、4个墙面安全净空，总计61维。新增字段放在末尾三个任务字段之前；原地图/目标/速度字段及末尾y、z、time按名称投影后逐值保持。默认仍为obs-v5，旧版本可按各自版本重放，旧checkpoint不得作为61维网络的warm-start。

净空为“原始图元有符号距离−本环境实际inflation_radius_m”，再除6m并裁剪到[-1,1]。零表示安全边界、正数表示仍有余地、负数表示已越界。full用r_fp+δ=3.119367799506481m，uncalibrated用r_fp=0.5055013879783843m；半径由每个环境显式传入，不取全局full默认值，也不重复膨胀图元。

16个方位仍按原始最近表面方位分箱，原始表面超过6m不加入，空箱为+1；保留原始障碍强度和mask。它是地图摘要，不能当作完整激光雷达、全局规划器或安全保护器。

实际消费者为训练的18地图环境、scalar/vector编码、actor及双critic、development构造器、原始样本审计和协调器。网络从显式契约获取61维；审计用独立NumPy几何计算检查obs与next_obs，拒绝错半径/错符号/旧维度，明确记录float32位姿导致的角度分箱边界不确定性。物理共用编码器仅通过合成状态数值一致性检查；真实物理obs-v7接线仍属于NAV-067。

本候选不变更h/y、预算扣减、共同停止、三套GAE、min-PPO、联合gamma=1、lambda=.95、动作初始化和优化超参。45秒任务、冻结地图/划分/预算/成本/阈值及最后checkpoint规则不变；v2整套冻结资源、旧校准及只读参考和论文保持原样。

## 本次验证与保留证据

- 最终门禁`reports/gate_clearance_feedback_v1_attempt4.json`：153/153通过，包含重跑的120项时间/反馈/生产门禁、73548条历史train重算、独立几何oracle、旧编码250组逐值对照、36个训练及12个development真实构造器（后者零rollout步）。
- 实际名义环境从墙面安全边界内2.5cm开始：向外/向内各走一步，分别产生负/正净空和有/无安全历史。沿真实close_segment和MinimizationUpdater的受控测试中，越界动作对安全动作的相对密度由0.8413降至0.7082；这是两动作相对密度，不是整个越界事件的积分概率。
- 原32套回归全部通过。最初自测中的测试接口调用、裁剪比较和分布属性错误已纠正，失败日志及中间门禁保留；未降低验收标准。
- 固定资源29个包文件及4个资产文件散列核验通过。旧源码、契约、预登记、原始报告和失败已保存；完整保全结果见`reports/nav064_clearance_feedback_release_v1.json`。

执行固定tiger11/tmux `nav064-r14-repair-20261115`，cwd `/home/caohy/repositories/Paper`，解释器 `/pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python`。沿用用户已批准的当前tiger11同机执行例外，无需再次请求SSH例外。外部快照与日志根为`/pub/data/caohy/work/camra_navigation/nav064_clearance_feedback_repair_20261116_v1/`；开工清单在`preflight.json`，回归在`regression/report.json`。原始tmux日志`/tmp/nav_clearance_*_20261116.log`在交付时镜像到该根目录的`execution_logs/`。

复算入口（在规定tmux内执行，输出必须换独立新路径）：

```bash
/pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python -u CaMRA/navigation/tests/test_clearance_feedback_gate.py --out <new_gate_report.json>
/pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python -u CaMRA/navigation/scripts/diagnose_clearance_feedback.py --out <new_diagnosis.json>
```

## 下一步与资源边界

待审提案`config/training_loop_clearance_feedback_proposal_v1.json`已绑定当前代码、契约和门禁，`training_approved=false`，未创建冻结pin或启动训练。具体追加量为两方法×三种子=6个fresh-init run，各600更新，18×227样本/更新，总14709600样本；首100包含在600内，做前缀审计后继续。只评估最后checkpoint，development为6图×3种子×2方法×3预算=108回合，无额外性能smoke。

上一轮次数已全部用完；计划`uncertainty_policy`明确要求改变正式试验次数先确认。确认后另建不可变预登记与pin，再核验资源/输出重名并启动，不覆写本提案及任何旧实验。没有自动重跑或扩额；训练/开发仍按原关闭档判定，未通过则保留失败并保持NAV-064=false。

先验证该观测候选，再依据结果处理剩余的超时、长期价值传递和随机训练/确定性部署差异。当前不同时调gamma/lambda或改目标；部署actor固定后的双critic重拟合与价值预算搜索分别留在NAV-065/066，不能提前冒充完成。

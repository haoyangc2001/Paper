# NAV-064：训练退化与已知终点尾项回修

本次是计划1.3.1的NAV-064回修。当前51/67项历史完成，NAV-064仍false；固定v2、原关闭档和下游依赖不变。正式新训练尚未批准或启动。结果入口为[实现验收报告](../reports/nav064_terminal_tail_repair_v1.json)、[新版门禁](../reports/gate_terminal_tail_v1_attempt2.json)和[未批准训练提案](../config/training_loop_terminal_tail_proposal_v1.json)。

## 已确认的问题与边界

上一轮avoidance-prior-v2已FINISHED，六组各300更新、11517556样本、32400完整训练回合及108开发回合。full三个种子的主成功率均66.7%，危险步0；开发L2全超时，结束仍剩49–78J。full L2训练首50更新900/900安全到达，末50降到277/900。同一批固定观测的u50→u300前进命令下降，存在真实训练后退化，不能解释成地图变难。

功率模型使直线每米能耗在约0.31m/s最省，初始约0.465m/s适当降低有理由；过度降速则错过45秒期限。超时的终点反馈在现有归一化minGAE中很弱：λ=.999、450步、无内部停止时，起点对最后边界的直接敏感度为λ^449/Σλ^k≈0.00176026。这个量是控制其他值不变的导数，不是“终点总奖励占比”，有内部停止时也不能套用该数。

旧train中预先指定的末50更新、full L2、k≥1.15超时轨迹共157条，真实停止残差全部为正，但127条起点联合target为负。终点传播弱与内部critic乐观共同构成可复现的退化机制线索；单条随机失败的负target不能单独证明critic错误，也不能把所有退化归因于一个超参数。熵梯度约1%、PPO符号及加性成本检查未支持其为主因。

## 与RC-PPO的关系

参考代码`RC-PPO/rl/gae.py:195`的归一化递推本身就采用当前窗口权重；本次不宣称发现参考实现错误。论文的特定actor-critic收敛结论也不保证本项目有限样本PPO每轮改进。

新版`rcppo-terminal-tail-v5`保持预算增广、安全历史、共同停止、双critic、三估计器、min Bellman及最小化PPO。针对已明确结束的有限任务，把剩余的几何权重交给已知终点：

- 剩余n步，k<n的k步目标权重为`(1−λ)λ^(k−1)`。
- 最后n步目标权重为`λ^(n−1)`，所有权重和为1。
- 仅真正任务结束使用；外部采样切片保留旧归一化混合及实际critic bootstrap，内部共同停止语义不改。
- reach及joint采用此尾项；costGAE、λ=.999、gamma_joint=gamma_cost=1、reach折扣日程均不变。

这是有独立契约的有限任务估计器适配，不能声称与参考代码逐位相同。gamma_joint=1保留本项目有限时域的零阈值目标；成本不折扣本身并不要求联合折扣必须为1。gamma=.999只作数值对照，不作为本轮新训练变量。更强终点权重可能增加估计方差，需正式学习检验。

上述450步条件下，新版直接终点敏感度为0.63812273。用同一157条旧轨迹、同一旧critic只重算权重，三种子的均值偏差均缩小，负target由127减到95；95条仍然存在，因此数值改善不能充当新策略成功证据。

## 已执行的验收

- 独立枚举352个短轨迹公式对照（含不同λ/γ、所有内部停止、失败及切片），最大误差2.22e−16；另验450步终点导数。
- 到达、超时、停滞、碰撞、越界、外部切片和安全/超支评分检查通过；共同停止及cost目标与v4逐位相同，旧157条轨迹全部目标可重放。
- 真正训练入口完成两次当前策略PPO更新，18图×2=36完整回合、11618样本；实际审计重算全部36块，old_log_prob及critic一致，拒绝篡改样本数、缺回合和陈旧更新标识。此为机械夹具，不作性能采样或候选选择。
- 累计452条门禁检查通过，209个不同检查名；包含父门禁重复项，不能称452次独立试验。32套旧版回归均通过。
- 第150更新稳定性检查通过合成原始库测试；真实协调器流程在模拟进程下验证50/150/300顺序、退化HELD、保留日志且不执行后续训练/开发。未批准提案被启动验证拒绝；只在内存置批准时全部实际绑定验证通过，没有保存批准或启动正式run。
- 284份旧源码/配置快照可校验；33项固定v2资源身份、16项旧结果/模型/日志hash保持不变。参考工程、论文及校准未改。

旧门禁保留为旧版本证据；新增runner/validation后，首轮父门禁hash失效，已使用新run名完成整链重验。一次复验因旧机械run名冲突被拒、一次因父报告尚未完成而未启动，均保留失败现场。有效父证据为工作根`parent_gates_attempt3/`，有效新版为`gate_terminal_tail_v1_attempt2.json`；不更新旧报告hash冒充复验。

## 下一次正式训练的具体提案

额外申请一次两方法（uncalibrated/full）×三种子，各300完整回合更新；前50和150均计入300。上限14580000样本；仅300更新最后checkpoint的108回合development，保持原预算网格及关闭档，不选最好checkpoint、不增加smoke或自动重试。采用独立`nav064_terminal_tail_v1`前缀，不挪用上一轮额度。

50更新保留原始审计和每个full种子安全L2探索前提。150更新先审计，再比较所有六张train L2图上k≥1.15的首50与第100–149更新：每个full种子两窗口各至少20回合，安全且预算内到达率均≥80%，下降≤10个百分点；任一失败则HELD_NO_RETRY。全体L2分母和失败同时报告，该条件子组仅作额外停机机制检查，不替换全业务成功率或正式开发验收。条件不足也不放行。

提案`training_approved=false`且无pin。用户确认额外配额后才另建批准登记/pin，复查tiger11、v2资源、源码绑定和输出冲突并提交后启动。本次实现门禁不代表NAV-064通过；正式训练/开发、稳定性及数据审计全部通过前，不执行NAV-065/066/067或物理验收。

## 执行与恢复入口

主机`tiger11.ios.ac.cn`，沿用已批准同机tmux例外；会话`nav064-r14-repair-20261115`，窗口`terminal-tail-*`。cwd为`/home/caohy/repositories/Paper/CaMRA/navigation`，解释器`/pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python`。

工作根`/pub/data/caohy/work/camra_navigation/nav064_terminal_tail_repair_20261117_v1/`保存preflight、source_before/after、plan_before、诊断/probe的失败与修正版本、regression、父门禁各attempt、新版机械run与审计、closecheck及日志。核心门禁命令：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python -u tests/test_terminal_tail_gate.py --out reports/gate_terminal_tail_v1_attempt2.json --work /pub/data/caohy/work/camra_navigation/nav064_terminal_tail_repair_20261117_v1/tail_gate_attempt2_work --base /pub/data/caohy/work/camra_navigation/nav064_terminal_tail_repair_20261117_v1/parent_gates_attempt3/prior_v2.json
```

该命令已执行，输出和run存在，不能原样重跑覆盖。恢复先核对当前计划、release及既有会话/进程/完成标记；正式新训练未启动，不恢复旧FINISHED或挪用旧配额。train/development为唯一来源，budget_selection与final封存。

## 后续记录：用户已批准新额度

用户随后批准上述独立6×300及108开发方案。原提案和原历史结论不改；批准登记为`config/training_loop_prereg_terminal_tail_v1.json`，pin `6ba8f849d9487cbaf1141f60ff0228997a03bb9f0e681909741a8d187ca8feb4`。主机、64份源码、33项固定资源、16旧结果与原门禁/32回归证据核验通过，见`reports/nav064_terminal_tail_training_release_v1.json`。新的launch工作根为`/pub/data/caohy/work/camra_navigation/nav064_terminal_tail_launch_20261117_v1/`；启动前提交，正式运行以其launch.json及协调器原始日志为准。NAV-064仍false，启动通过不能代替学习与开发验收。

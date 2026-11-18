# NAV-065 v8实际结果与主agent复核

用户确认v8后在tiger11/tmux完成一次固定full_s0单候选验证，legacy主agent验收，无测试子agent。复用该候选162条train/59877步，新增回合/开发评分0；实际数值300+边界0、共300更新。临界数据到达相对MAE 80.35%→43.85%（原线35%），Pearson 0.3312→0.5722（原线0.70）；起点误放行3→13、误拒绝10→1。停止原因numeric_gate_not_met，pilot完整train FAIL；六actor及其余五模型不变、33项v2身份匹配。无新G4/部署包，原G4 FAIL、52/67、NAV-065=false及NAV-066未启动保持；本次单次尝试已结束，不自动扩展或挪用阶段余量。

| 项目 | 开始 | 末次 | 原要求 |
| --- | --- | --- | --- |
| 临界数据reach相对MAE | 80.35% | 43.85% | ≤35% |
| 临界数据reach Pearson | 0.3312 | 0.5722 | ≥0.70 |
| 162起点错误放行 | 3 | 13 | 0 |
| 162起点错误拒绝 | 10 | 1 | 按原主预算/边界标准 |
| 所选边界符号错判 | 97 | 97 | 0 |

数值门槛未全部通过时禁止进入边界纠错；本次结果应按实际停止原因解释。整体平均误差下降与起点符号可靠性是不同要求，不能以单项改善代替完整验收。

本轮只改critic，固定actor没有更新，因此这些误判变化不能解释为机器狗导航动作退化。NAV-064既有主预算名义导航结果保留；本次没有新的开发、物理或预算搜索成绩。若本轮未进入边界阶段，也没有实际证据证明受限SGD纠错有效或无效。

生产调用链保持：固定v7末次来源→原共同停止标签→原分层数值拟合/有条件边界纠错→每步原v5 train_gate→独立保存及CPU重载；本pilot无开发评估调用。原W=max(V_h,V_c−z/Z_REF)、beta=0、网络/obs-v7、J单位、数据用途及固定v2配套配置保持。

主agent复核批准pin与快照、57项基本测试的对应源码hash、162条原始轨迹/J账本/共同停止、全部逐步起点算术/标签/分类变化、阶段切换/额度及阶段/末次模型CPU重载。其余五来源模型全部张量及六actor保持，33项v2资源hash匹配。中间日志已核对，未重放优化器；阶段外没有每步模型张量。执行完整性通过不等于科学通过。

末次完整train失败项：

- false feasible budget at episode start
- vh additional relative MAE
- vh additional Pearson
- additional false feasible starts
- selected boundary row signs

后续先依据本轮已有train证据分析数值拟合的剩余误差和边界表现，再提出必要的有界改动。不得自动追加步数、转移阶段余量、挑中间模型、降低门槛或提前进入NAV-066。

证据：[主agent复核](../reports/nav065_pilot_refit_v8_review.json)、[批准登记](../config/nav065_pilot_refit_prereg_v8.json)、[原方案](nav065_pilot_refit_v8_20261118.md)。输出`/pub/data/caohy/work/camra_navigation/nav065_pilot_refit_v8/`；资源/源码快照/原始复算`/pub/data/caohy/work/camra_navigation/nav065_pilot_refit_launch_20261118_v8/`；tmux `nav064-r14-repair-20261115:nav065-v8-run/nav065-v8-acceptance`，Python `/pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python`。

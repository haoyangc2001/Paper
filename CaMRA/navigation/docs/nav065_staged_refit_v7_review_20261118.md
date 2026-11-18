# NAV-065 v7实际结果与主agent复核

用户确认v7方案后在tiger11/tmux完成一次独立两阶段critic回修，legacy主agent验收，无测试子agent。复用972条train/372471步，新增回合0；六候选数值+边界步数分别150+150/150+150/150+150/150+150/150+150/150+150，共1800次更新，六actor不变、33项v2身份匹配。起点误放行25→19，所选边界错判932→573；旧五档4/6、完整train 0/6通过，状态HELD_TRAIN_GATE。按批准协议开发重评0，无新G4/部署包；原G4 FAIL、52/67、NAV-065=false、NAV-066未启动保持。本次额度已结束，不能自动重启或转移余量。

| 候选 | 数值+边界更新 | 起点误放行（前→后） | 新增reach相对MAE（前→数值段→末次） | 边界错判（前→后） | 完整train |
| --- | --- | --- | --- | --- | --- |
| uncalibrated_s0 | 150+150 | 2→3 | 73.7%→52.6%→90.7% | 55→51 | FAIL |
| uncalibrated_s1 | 150+150 | 3→3 | 89.7%→53.7%→82.7% | 79→96 | FAIL |
| uncalibrated_s2 | 150+150 | 6→1 | 81.4%→55.3%→85.9% | 91→80 | FAIL |
| full_s0 | 150+150 | 5→3 | 63.8%→43.4%→80.4% | 96→97 | FAIL |
| full_s1 | 150+150 | 4→2 | 78.8%→45.8%→77.3% | 237→140 | FAIL |
| full_s2 | 150+150 | 5→7 | 100.2%→46.5%→75.0% | 374→109 | FAIL |

顺序为无校准seed0/1/2、完整方法seed0/1/2。相对MAE分母和旧/新分片保持原标准，门槛为35%；原始数据和每阶段模型保留。

本轮取消了旧正确起点的过程单调保护，实际优化每步都执行，没有按旧保护回退。逐步日志区分误放行、误拒绝及主预算变化；完整train每步检查，未达到全部标准。因此不能再仅用“保护锁住了更新”解释本轮失败；本次有限步数和分阶段方法仍不足以让原数值与符号条件同时满足。这不证明固定目标不存在可拟合解，也不证明只需继续加步数。

论文§4.2固定部署actor后重估价值、再做预算搜索的顺序保持；生产入口直接调用原v5 train_gate和原部署evaluate/G4。共同停止目标、beta=0、actor、v2/校准、地图、单位和分片均未改。本次未触发开发评估，因此不能推断开发性能变好或变差，也不影响NAV-064既有主预算名义策略结果；未证明物理闭环成功。

各候选末次失败项：

- uncalibrated_s0: vh additional relative MAE; vh additional Pearson; additional false feasible starts; selected boundary row signs
- uncalibrated_s1: vh additional relative MAE; vh additional Pearson; additional false feasible starts; selected boundary row signs
- uncalibrated_s2: vh additional relative MAE; vh additional Pearson; additional false feasible starts; selected boundary row signs
- full_s0: vh additional relative MAE; vh additional Pearson; additional false feasible starts; selected boundary row signs
- full_s1: false feasible budget at episode start; vh additional relative MAE; vh additional Pearson; additional false feasible starts; selected boundary row signs
- full_s2: vh relative MAE failed or undefined; vh additional relative MAE; vh additional Pearson; additional false feasible starts; selected boundary row signs

主agent复核了批准pin/源码快照、33项v2、972条原始轨迹/372471步的J账本与共同停止真值、每个更新的阶段及额度、逐起点分类变化、阶段模型、六actor和CPU重载。79项准备基本测试证据与未变源码匹配，本轮没有为凑测试数再次训练。逐步模型张量只在阶段边界保存；中间日志的数值/算术和分类变化已核对，未重放优化器。

后续先用现有train日志分析数值阶段和边界阶段的冲突、具体误判分量及数据表示；若需改变过程、网络或新增训练额度，另行形成可审阅方案。不能挑中间checkpoint冒充批准末次结果、放宽门槛、转用开发数据拟合或提前启动NAV-066。

证据：[机器复核](../reports/nav065_staged_refit_v7_review.json)、[批准登记](../config/nav065_staged_refit_prereg_v7.json)、[方案](nav065_staged_refit_v7_20261118.md)。外部输出`/pub/data/caohy/work/camra_navigation/nav065_staged_refit_v7/`；启动/资源/源码快照/逐例复算`/pub/data/caohy/work/camra_navigation/nav065_staged_refit_launch_20261118_v7/`。执行提交`3a4362a`，tmux `nav064-r14-repair-20261115:nav065-v7-run/nav065-v7-acceptance`；Python `/pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python`。

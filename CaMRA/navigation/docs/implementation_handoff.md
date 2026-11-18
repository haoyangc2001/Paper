# 上层导航实施交接

## 2026-11-18 NAV-065 v7准备完成：两阶段拟合、最终门禁保持

主agent按legacy完成隔离v7入口、79项基本测试和972条train/372471步只读预检，无测试子agent。六份v6源模型的实际train门禁重算一致，33项v2身份匹配，正式optimizer/新回合/development重评均0。v7先数值回归、再沿用边界符号梯度纠错；取消离线过程单调保护，每步记录全部起点预测及误放行/误拒绝变化，原目标、actor、beta、最终v5 train/G4和完整v2保持。

v6拒绝原因23次旧正确起点保护、4次旧五档门禁；没有中间逐起点预测和checkpoint，不能恢复具体被牺牲的起点或虚构全部是保守拒绝，本轮没有重放真实optimizer。25次原误放行仍保留。原G4 FAIL、NAV-065=false、52/67、NAV-066未启动。

[具体提案](../config/nav065_staged_refit_proposal_v7.json)未批准：每候选最多150步数值+150步边界、总1800新试步，新增回合0；全train通过后才原180 development重评一次。过程规则和数字额度需明确确认，不能重用已结束v6余量。详见[v7方案](nav065_staged_refit_v7_20261118.md)、[真实预检](../reports/nav065_staged_refit_preflight_v7.json)及[准备复核](../reports/nav065_staged_refit_v7_preparation_review.json)。下一步为确认后冻结批准登记/hash、现场再核资源、提交并运行一次新目录；本轮仅准备，以下为历史时点。

## 2026-11-18 NAV-065 v6结束：短段离线回修仍未过完整train

用户确认v6方案后在tiger11/tmux完成一次独立离线回修。复用972条train/372471步，新增回合0；六候选试步16/16/16/28/16/24，共116次，接受更新0/0/0/4/0/4，共8次，六actor不变。旧五档train仍6/6通过；新增临界预算误放行27→25，所选边界错判981→932，完整train 0/6通过，状态HELD_TRAIN_GATE。按批准失败协议，开发重评0、无新G4/部署包；原G4 FAIL、52/67、NAV-065=false和NAV-066未启动保持。未用试步不能自动续跑、转移或重启本run。

主agent按legacy完成63基本测试、972条轨迹/J/共同停止及实际门禁、来源/33项v2、段计量与CPU重载核验，无专用测试子agent。见[v6详细结果](nav065_block_refit_v6_review_20261118.md)及[机器报告](../reports/nav065_block_refit_v6_review.json)。产物与源码快照独立保存；下一步先分析现有train失败，不能自动续跑。下方为历史时点。

## 2026-11-18 NAV-065 v6准备完成：972条train预检、63项基本测试通过，未正式拟合

用户指定legacy继续推进，由主agent基本验证，无测试子agent。新版隔离入口按旧/新、起点/沿途/边界及到达目标正负分层回归；拟每4步一段、源模型锚定工作波动边界，段末保护旧结果/正确起点，最终仍直接调用原v5 train_gate及原G4。63项基本测试通过，972条train/372471步的来源、J/共同停止、预算顺序及固定actor实际动作预检通过；只读梯度诊断进一步支持回归/符号方向冲突。

正式采样/拟合/development重评均0。[v6提案](../config/nav065_block_refit_proposal_v6.json)请求每候选最多300新试步/总1800，新增回合0，六组完整train全过后才原180 development重评一次；工作保护改动与数字额度尚待明确确认，不能重用v5余量。原目标、beta、最终标准、actor及v2不变。NAV-065=false、52/67、原G4 FAIL和NAV-066未启动保持。详见[v6说明](nav065_block_refit_v6_20261118.md)及[真实预检](../reports/nav065_block_refit_preflight_v6.json)。下方为历史时点。

## 2026-11-18 NAV-065 v5执行结束：补采432回合，仍HELD，未重评开发集

用户最新确认后按独立批准登记执行一次v5，固定tiger11/tmux、legacy主agent验证，无专用测试子agent。新增432条train/165173步，连同旧540条共972条/372471步；六候选试更新28/16/24/8/8/8，共92次，接受7/1/4/0/0/0，共12次，六actor不变。旧五档train仍6/6通过，但新增临界预算起点误放行43→27，新增到达critic相对MAE约64%–100%，六组均未通过完整train检查，状态HELD_TRAIN_GATE。按协议开发重评0，无新G4/部署包；原G4 FAIL、NAV-065=false、52/67、NAV-066未启动。未用额度不自动续跑或转移。主agent复核972条轨迹/J/共同停止、采样顺序、试步计数、CPU重载及源码身份通过；基本48测试通过不等于科学通过。

见[本轮结果](nav065_adaptive_refit_v5_review_20261118.md)及[复核报告](../reports/nav065_adaptive_refit_v5_review.json)。正式输出`/pub/data/caohy/work/camra_navigation/nav065_adaptive_refit_v5/`，启动/源码快照/复核`nav065_adaptive_refit_launch_20261118_v5/`。下方为历史时点。

## 2026-11-18 NAV-065 v5准备完成：train边界覆盖已核验，新额度待确认

按用户指定legacy由主agent验证，不派专用测试子agent。复核540条train/207298步及六份v4固定actor模型：每候选18张图均有预算可行性转换，旧五档的相邻交界宽度全为k=0.15；abs(W真值)≤0.01的起点共49/540。这支持有限临界补采，不证明全部误判唯一根因或保证新一轮通过。

[隔离v5入口](../scripts/refit_nav065_adaptive.py)和[未批准提案](../config/nav065_adaptive_refit_proposal_v5.json)已备妥：每个actor×train图加4个自适应预算，共最多432新train回合/194400步；只拟合双critic，每候选最多300次试步、总1800，拒绝试步也计入。新增预算必须真实重跑固定actor；分量及联合W符号拟合覆盖起点与沿途临界状态。原目标、beta、G4与v2不改，六组train全过才统一重评原180 development一次。新数据和数字额度未批准，未创建正式run；不能挪用旧剩余额度。

48项基础检查及只读覆盖诊断通过，实际采样/优化/development重评均0，失败工程attempt也保留。详见[说明与验收边界](nav065_adaptive_refit_v5_20261118.md)及[有效诊断](../reports/nav065_train_coverage_v5_attempt3.json)。原G4 FAIL、NAV-065=false、52/67、NAV-066未启动保持；下一步仅在新额度明确确认后冻结批准登记、再核验资源并实际执行。下方为历史时点。

## 2026-11-18 NAV-065 v4结束：train6/6，开发G4仍FAIL

用户允许新一轮测试，legacy主agent验证、不派子agent。v4仅full seed1最多300次试更新，实际1次1e-5后train全过、错误放行1→0；其他五模型及六actor不变。复用540 train/207298步，新增回合0；统一复评180旧development一次，主预算36/36正确，但紧预算4次误放行，仅2/6候选通过，新G4 FAIL、无部署包。31项基础测试及720条原始轨迹/J/共同停止复核通过。NAV-065=false、52/67、NAV-066不执行；按失败协议停止，剩余299次不自动使用。

见[详细结果](nav065_guarded_refit_v4_review_20261118.md)、[批准登记](../config/nav065_guarded_refit_prereg_v4.json)及[主agent报告](../reports/nav065_guarded_refit_v4_review.json)。产物位于`/pub/data/caohy/work/camra_navigation/nav065_guarded_refit_v4/`，启动/资源/源码/复核在`nav065_guarded_refit_launch_20261118_v4/`。下一步先分析train预算边界覆盖与跨地图识别，不能把开发失败直接转train；新拟合须独立方案与授权。下方均为历史记录。

## 2026-11-18 NAV-065边界回修v3已结束：5/6 train通过，仍HELD

按用户确认的[独立登记](../config/nav065_boundary_refit_prereg_v3.json)执行一次，16:07:20–16:08:12在tiger11/tmux完成。四组零更新保留，full seed1更新300、seed2首次10更新通过，共310；六组actor不变。540条train/207298步复用，新增回合0。主agent按legacy直接复核，未派专用测试子agent。

假可行4→1，假不可行2→3（非主档诊断）；full seed1最终均值误差达标，仍在L1-s1004/k1.0将超支2.8754 J的安全到达误判为可行。中段零误放行时平均误差未达标，不能挑选回填。full seed2已接受17个真实可行主档并拒绝真实超时，策略失败未改写为成功。整体HELD_TRAIN_GATE，未开发重评、无新G4/部署包，原G4 FAIL、52/67、NAV-065=false及NAV-066未启动保持。

见[详细结果](nav065_boundary_refit_v3_review_20261118.md)及[主agent报告](../reports/nav065_boundary_refit_v3_review.json)。正式产物`/pub/data/caohy/work/camra_navigation/nav065_boundary_refit_v3/`；启动/源快照/登记/逐例复算在`nav065_boundary_refit_launch_20261118_v3/`。full seed1的300上限已耗尽，不转移其他候选未用额度、不重启本run；后续先以现有train证据分析损失冲突。下方未批准与待启动记录均为历史时点。

## 2026-11-18 NAV-065边界回修v3准备完成，尚未正式拟合

按用户指定legacy技能，由主agent基础检查、不安排测试子agent。新版入口保持actor、共同停止目标及beta=0，在critic回归中增加边界分量符号纠错；train主档按真实可行数检查，full seed2接受17并拒绝超时，五档零假可行不变，development原6/6及G4门槛不变。22项基础测试通过，540条train/207298步实际动作与真值预检通过；四组可零更新保留，full seed1/seed2各两次假可行仍未解决。

[详细说明](nav065_boundary_refit_v3_20261118.md)、[未批准提案](../config/nav065_boundary_refit_proposal_v3.json)、[预检报告](../reports/nav065_boundary_refit_preflight_v3.json)已就绪。拟两失败候选各最多300、总600新critic更新，其余四组零更新，新增回合0；六组train全过才复验180条development一次。本轮正式更新0、开发重评0，未创建正式run。新数字额度待确认，不能重用或转移旧额度。52/67、NAV-065=false、原G4 FAIL、NAV-066未启动。以下记录均保留为历史时点。

## 2026-11-18 NAV-065离线重拟合v2已结束，4/6 train通过，整体仍HELD

用户确认后，按[批准登记v2](../config/nav065_saved_refit_prereg_v2.json)执行一次有限离线重拟合。新增采样0，复用540条train，六候选新增更新分别10/10/10/10/300/300，合计640；四组首次train检查通过即冻结，full seed1/seed2到各自300上限仍失败。运行状态`HELD_TRAIN_GATE`，未评分development、未产新G4/部署包。原G4 FAIL有效；NAV-065=false、52/67，NAV-066及下游未启动。不重启本run或挪用余下总额度。

主agent按用户免除专用测试子agent的要求完成540条轨迹复算、CPU重载、actor逐tensor一致、更新序列/早停、原输入及33项v2生产预检。train起点假可行13→4，剩3次紧预算超支放行和1次主预算超时放行。新发现原full seed2在train主档实际只有17/18可行：L2-s1006在45秒时距目标0.577576 m（目标半径0.5 m）、仍余31.103 J。批准方案未先核对此事实就要求train 18/18，导致该候选门槛在固定actor条件下不可达；门槛和失败保留，不改标准回填通过。这与NAV-064原development主档36/36是不同地图分母。

见[详细结果及方案检查遗漏](nav065_saved_refit_v2_review_20261118.md)和[完整复算报告](../reports/nav065_saved_refit_v2_review.json)。两个失败候选的新增更新上限已耗尽；后续先评审门槛适用性及边界符号拟合方案，新增更新/改变门槛须明确授权。下方“提案未批准”等均为历史时点，原提案false保留、实际执行使用独立批准登记。

## 2026-11-18 NAV-065只读边界诊断完成，离线回修入口已备妥

用户要求后续不再安排专用测试子agent；本轮由主agent直接核验，不伪称独立。复用540 train及180 development原始轨迹和拟合前后权重，新增回合/优化更新均为0。train起点假可行3→13、development 3→9，尽管成本起点MAE分别7.46→1.59 J、5.94→2.00 J。问题已出现在训练图，不能只归因于新地图泛化；旧模型的成本高估偏差被消除后，边界错误放行反而增多。九次开发误判均低估共同停止成本，八次到达评价也把共同停止前的小正数预测成负数。见[诊断与下一步](nav065_boundary_diagnosis_20261118.md)和[逐回合证据](../reports/nav065_boundary_diagnosis_v1.json)。

[离线回修入口](../scripts/refit_nav065_saved.py)和[待批准提案](../config/nav065_saved_refit_proposal_v2.json)已建立：新增回合0，仅复用train做双critic起点/沿途/边界回归，每候选最多额外300更新、总计最多1800；先过六组train检查再统一开发复验一次，原目标、beta=0及G4标准不改。提案`approved=false`，尚未启动新拟合。NAV-065仍false、NAV-066未启动；新的更新额度须确认后才能执行。

旧重拟合审阅的训练日志SHA256是抄录错误，原报告保留；诊断报告追加正确值及候选清单的运行时散列依据。原始540条训练数据未被改写。下方为各历史时点记录。

## 2026-11-18 NAV-065固定策略双critic重拟合完成，G4仍未通过

按冻结登记[重拟合预登记](../config/nav065_refit_prereg_v1.json)完成最多540个train名义回合：固定六个actor，每候选90次critic更新，仅拟合成本与到达critic。540条训练回合覆盖18张train图和五档预算，共207,298个轨迹步；开发侧复用全部180个既有development回合，没有新增开发回合。33项固定v2输入核验通过，预算输入裁剪为0，六个actor张量保持不变，六组双critic均有更新。

独立轨迹复算确认主预算`k=1.15`的36/36个可行起点均判断正确，六组平均误差指标均达标；但紧预算发生9次假可行（`k=0.85`为3次，`k=1.00`为6次），实际均安全到达但超过剩余预算0.56–7.65 J。每候选零假可行的冻结标准未达到，六候选0/6通过，G4为`FAIL`，未生成部署候选包。NAV-065保持false，NAV-066及下游不能启动。

测试员R1的原始轨迹/J/W交叉核验通过，但因未运行代码级变异测试而标为blocked；R2随后在隔离副本禁用重复/未知development回合保护，目标断言按预期失败，恢复后通过。R2的流程测试已覆盖，独立报告仍按科学验收结果判fail：G4失败不因测试覆盖补齐而变成通过。详细报告见[重拟合审阅](../reports/nav065_refit_v1_review.json)、[R1](../../../.worklog-testing/NAV-065/NAV-065-R1/report.json)和[R2](../../../.worklog-testing/NAV-065/NAV-065-R2/report.json)；原始训练及复验产物在`/pub/data/caohy/work/camra_navigation/nav065_refit_v1/`。本轮540回合额度已用完；没有新的训练额度或改变beta、验收线的授权。后续须先基于现有证据制定NAV-065合规的下一步方案并取得所需资源授权，G4通过前不得进入NAV-066。

## 2026-11-18 NAV-065 五档正式评估完成，G4 因三次预算误判未放行

提前冻结 [NAV-065 标准](../config/nav065_deployment_prereg_v1.json)：`beta=0`、每候选零假可行、主预算 `k=1.15` 的六个可行起点全部识别，沿用原误差标准。tiger11/tmux 正式评估及只读复核均已完成，计 108 个复用回合、72 个新增回合，共 180 个。主档 36/36 安全且预算内到达并判断正确；紧档三次假可行均为安全到达但超预算 2.10～2.82 J。六个候选均通过平均误差项，三个候选未过假可行门槛，未发布候选包。

[复核报告](../reports/nav065_deployment_v1_review.json)索引原始回合、正式报告和 G4 FAIL 的散列；[结果与重拟合提案](nav065_deployment_review_20261118.md)说明下一步拟固定 actor、只重拟合双评价网络，以及拟新增的 540 个 train 名义回合。依据当前计划的资源约束，新增额度尚未确认。NAV-064 保持完成，NAV-065 保持未完成，NAV-066 及后续未启动。下方“等待评估授权”均为正式评估前的历史状态。

## 2026-11-18 NAV-065 启动前检查通过，等待评估授权

[评估入口](../scripts/evaluate_nav065_deployment.py)会在消耗新额度前核对六份模型散列、108条旧轨迹的起始预算和五档预算的观测分辨率。实际只读检查通过：六候选、108旧回合、60个方法×地图×预算组合；未加载模型、未产生预测或新回合。

正式运行在独立目录保留逐回合记录、双评价报告和 G4 审阅件；覆盖不足或任一候选失败时不生成候选包。18项合成测试通过，审阅件不会改写计划状态或历史产物。[提案](nav065_deployment_validation_proposal_20261118.md)尚未获确认，无正式评估登记；NAV-065仍未完成。

## 2026-11-18 NAV-065 正式评价入口已实现，尚未放行

[评价入口](../scripts/evaluate_nav065_deployment.py)校验单独批准且带 SHA256 的登记、固定候选、上游门禁和 development 来源，再按预算逐回合保存轨迹及预测。初始误判计数和逐步误差分别统计；缺回合或重复回合不能通过。未批准时命令退出并且不建运行目录。轨迹及汇总共17项合成测试通过；未执行真实评价或产生 G4 结果。

恢复时按[提案](nav065_deployment_validation_proposal_20261118.md)先确认回合数及门槛，再冻结 `config/nav065_deployment_prereg_v1.json` 和 SHA256。运行目录由登记绑定，已存在的目录不得覆盖；中断后先查 `episodes.jsonl` 中的开始和完成记录。只有实际预测、覆盖、G4 和候选交付全部验收后才能完成 NAV-065。

## 2026-11-18 NAV-065 独立轨迹判定已准备

以完整轨迹独立判定安全、到达、预算，用同一停止点计算双评价对照目标，不用网络给自己提供真值。吸收失败强制取终点并保留末步成本替换规则；超时和停滞用实测终态。对照目标是确定性全轨迹值，训练中的折扣近似和 λ 平滑不参与。分别核对动作前观测与动作后记录，预算判据用未裁剪的有符号 J 除以固定尺度。

[兼容性审计](../reports/nav065_saved_trajectory_compatibility_v1.json)复核了33项冻结 v2 文件、108条已验收轨迹及42187个动作前观测，未发现预算特征裁剪。合成反例通过。未加载评价网络、未新增回合；当前52/67，NAV-065仍需确认并冻结[验收协议](nav065_deployment_validation_proposal_20261118.md)，再执行双评价检查。G4 和 NAV-066 均未放行。

## 2026-11-18 NAV-065 已绑定来源，等待冻结部署评价协议

六份末次模型已在 tiger11/tmux 上严格 CPU 重载，42,187 个开发动作复算一致，最大绝对误差约 9e-8。候选来源、策略与双评价权重散列见[核验报告](../reports/nav065_deployment_source_binding_v1.json)。[验收提案](nav065_deployment_validation_proposal_20261118.md)列明独立目标、误判放行线和补齐两档边界预算所需的最多72个开发回合。未读取新部署价值结果或启动新评估；NAV-065尚未验收，NAV-066及后续未执行。

## 2026-11-18 NAV-064 正式结项，六份最终模型交给 NAV-065

独立验收用保存的轨迹复算108个开发回合，绑定了六份末次模型、18张train图及6张development图的实际输入散列、当前生产代码和冻结v2身份。总计1800更新、32400完整训练回合、11136866样本，50/150/300原始审计及探索/稳定性均通过。主预算两方法各18/18安全且预算内到达，完整方法L2为6/6、危险步0。完整方法紧档2/18只作预算响应诊断。

以[验收报告](../reports/nav064_terminal_tail_acceptance_v1.json)为下一阶段唯一候选来源。计划52/67、NAV-064=true；NAV-065固定六份末次actor，需验证匹配双评价网络及新G4，之后才能使用独立选择集和进行物理验证。下方为历史记录，不续跑旧训练。

## 2026-11-17 终点尾项v5已实际启动，首50审计和探索通过

六组于21:30在`tiger11.ios.ac.cn`、`nav064-r14-repair-20261115:terminal-tail-training`启动，提交`6b35db96d9d84dabc08e4b7e4c767409559a78f2`，批准登记pin为`6ba8f849d9487cbaf1141f60ff0228997a03bb9f0e681909741a8d187ca8feb4`。解释器`/pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python`，cwd为导航目录；独立核验实际六进程、当前源码/33项v2身份、输入hash、完整回合及多epoch更新，见`reports/nav064_terminal_tail_v1_u50_v1.json`。核验快照更新数为71/60/77/59/58/70，共2257617样本；这是当时进度，恢复须查最新日志。

首50原始审计PASS：5400完整回合、1714070样本、216块独立数值重算，元数据不一致0。full三个种子L2各300次全程安全到达，共900/900；其中预算内分别127/130/127，共384/900，保留随机紧预算和全分母。探索PASS只说明初期可绕障，不能替代后续稳定性或开发验收。

协调器正在按批准额度继续：150原始审计及train稳定性失败会HELD_NO_RETRY；通过才继续300、全量审计和最后checkpoint108开发。NAV-064=false，51/67历史完成，NAV-065+不执行。恢复先查`/pub/data/caohy/work/camra_navigation/nav064_terminal_tail_v1/{start,pids,HELD,FINISHED}.json`和launch根`nav064_terminal_tail_launch_20261117_v1/{launch.json,coordinator.log,verified_u50_attempt1/}`，禁止重启同run、转移额度或覆盖旧结果。下方状态均保留为历史时点。

## 2026-11-17 终点尾项v5额度已批准，准备实际启动

用户批准已评审的额外6×300完整回合更新、上限14580000样本及最后checkpoint108开发回合；50/150均计入300，原探索/稳定性/关闭档和失败保留规则不变。原proposal仍false，批准身份独立冻结为`config/training_loop_prereg_terminal_tail_v1.json`和pin `6ba8f849d9487cbaf1141f60ff0228997a03bb9f0e681909741a8d187ca8feb4`。本次不改源码、目标、评分、超参数、数据划分或冻结v2，不挪用旧额度。

tiger11沿用已批准同机tmux例外，会话`nav064-r14-repair-20261115`，Python `/pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python`。64份当前源码/配置、33项v2身份、16旧结果及452门禁/32回归原证据均核验，输出无冲突，原avoidance-v2 FINISHED且六个旧PID均不在采样。新放行见`reports/nav064_terminal_tail_training_release_v1.json`；工作根`/pub/data/caohy/work/camra_navigation/nav064_terminal_tail_launch_20261117_v1/`保存preflight、source_before、registration及training_release。

启动前本地提交，随后在`terminal-tail-training`窗口运行`run_feedback_round.py --prereg training_loop_prereg_terminal_tail_v1.json`。恢复先查launch.json、协调器`/pub/data/caohy/work/camra_navigation/nav064_terminal_tail_v1/{start,pids,HELD,FINISHED}.json`及原始日志，不重复启动。50/150/300原始审计与最后checkpoint108开发全部通过前NAV-064=false，51/67历史完成，NAV-065+不执行。实际启动须有独立核验证据，不能仅凭授权或进程存在判断。下文为历史时点。

## 2026-11-17 终点反馈回修v5：实现通过，NAV-064仍false

上一轮avoidance-prior-v2已FINISHED，六组300更新/11517556样本/32400回合，108开发FAIL_DEVELOPMENT；full三个种子主成功率66.7%、危险步0，L2均超时。首50→末50的full L2安全到达900/900→277/900，存在训练后退化。旧源码、模型、失败及日志保留，不按下方旧“正在训练”状态续跑。

已版本化为`rcppo-terminal-tail-v5`，仅真正任务结束的reach/joint minGAE保留几何尾项；共同停止、costGAE、外部切片、γ/λ、先验v2、任务/预算/地图/评分及冻结v2均保持。450步无内部停止的终点直接敏感度0.176%→63.8%；旧157条train超时目标中负值127→95，仍有缺口，不能据此宣称新学习通过。参考归一化实现不标成错误，本项明确登记为有限任务适配。

有效门禁`reports/gate_terminal_tail_v1_attempt2.json`为452条累计检查、209个不同名（有父检查重叠），含独立352公式用例、真实两次PPO更新/36完整回合/11618样本及独立审计；32套回归通过。实际协调器在模拟进程测试中验证150退化会HELD且不继续300/开发；原始合成库正常/退化/篡改分支验证通过。284份原快照、33项v2身份及16旧结果/模型/日志hash校验不变。详情、失败attempt、精确命令及边界见[回修说明](nav064_terminal_tail_repair_20261117.md)与`reports/nav064_terminal_tail_repair_v1.json`。

工作根`/pub/data/caohy/work/camra_navigation/nav064_terminal_tail_repair_20261117_v1/`；tiger11，原tmux会话`nav064-r14-repair-20261115`，Python绝对路径不变。有效当前源码父链在`parent_gates_attempt3/`，旧hash失配/旧run名冲突/父报告未就绪的attempt均保留，不覆盖或重写旧锁。新版正式前缀`nav064_terminal_tail_v1`尚无run。

未批准提案为`config/training_loop_terminal_tail_proposal_v1.json`，额外申请两方法×三种子×300完整回合更新，上限14580000样本、最后checkpoint108开发回合；前50及150计入300。150额外train稳定性条件为每个full种子k≥1.15的L2首50与100–149窗口各≥20回合、两窗口安全可行率≥80%且下降≤10个百分点；保留全分母，失败或不足HELD_NO_RETRY，不替代原关闭档。提案false、无pin；批准后另冻登记并复核现场/输出/源码后启动，不转移旧额度。当前51/67历史完成、NAV-064=false、NAV-065+不执行；新训练/开发/稳定性/数据验收均通过才进入后续任务。下方旧时点保留。

## 2026-11-17T17:12:13+08:00 避墙/选边v2已启动：PASS_STARTED

启动提交`b0a0ce9f56a72a37834ffa62538e231bbcab26eb`，冻结pin `daa57ee473e9cfbb2b4ad4c84780673156f4ee8058591729b07339ccd6df27c0`。窗口`nav064-r14-repair-20261115:avoidance-v2-training`在tiger11运行六组；独立核验时各2更新、合计67805样本。原始header/输入hash、18图、61维观测、v2先验、完整回合、10epoch×512及实际目标匹配，六进程正常且无HELD；证据`reports/nav064_avoidance_prior_v2_started_v1.json`及launch根`verified_startup_attempt1/`。

恢复先查既有launch/start/pids、原始日志、HELD/FINISHED和进程，不重复启动。协调器自动首50审计、逐full种子安全L2探索，通过后才继续300及最后checkpoint108开发验收；失败保留现场且不自动重试。启动核验不作学习结论；原关闭档未满足，NAV-064=false、NAV-065+未执行。下文均为历史时点。

## 2026-11-17 避墙/选边v2训练已获授权，待实际启动核验

当前NAV-064=false、历史51/67完成。用户已批准独立6×300完整回合更新、上限14580000样本及最后checkpoint的108开发回合。新登记`config/training_loop_prereg_avoidance_prior_v2.json`与pin `daa57ee473e9cfbb2b4ad4c84780673156f4ee8058591729b07339ccd6df27c0`已冻结；原proposal不改，旧v1 HELD、原始日志和六个u50模型保留，不恢复或转移额度。

219门禁、32回归、60份源码配置快照与33项固定v2身份复核通过，来源见`reports/nav064_avoidance_prior_v2_training_release_v1.json`。本次只登记/执行已批准训练，算法、先验v2源码、观测、预算/地图/45秒/评分及固定v2资源不变。新launch根`/pub/data/caohy/work/camra_navigation/nav064_avoidance_prior_launch_20261117_v2/`；协调器根`/pub/data/caohy/work/camra_navigation/nav064_avoidance_prior_v2/`。启动前本地提交，原tmux会话新增`avoidance-v2-training`窗口，绝对解释器不变。

实际启动证据以launch根`launch.json/coordinator.log`和协调器`start.json/pids.json`为准；未出现证据前不得认为已训练。前50计入300：原始审计通过且三个full种子各至少一次安全L2到达后才继续；否则自动HELD，无自动重试/补额。300全量审计及108开发验收仍须通过原标准；budget_selection/final封存，NAV-065+不执行。下文各段保留历史，不按旧启动状态续跑。

## 2026-11-17 NAV-064避墙/选边回修v2

当前51/67历史完成、NAV-064=false。旧avoidance-v1首50原始审计PASS、探索FAIL并HELD，实际348更新/1432145样本和六个u50模型保留，不能按下方旧“正在训练”描述恢复。确认的独立问题为上下墙法向错误和量化角零阈值选边振荡。

新clearance-tangent-v2保留v1复现分支，更正墙法向，按pi/8观测量化歧义固定顺时针选边；默认关闭网络、RC-PPO目标/评分/任务/数据/初始化与固定v2整套不变。64个四墙多航向检查通过；旧900条full L2观测不变重算，选边切换108036→0；一次36条seed0 train机制回合全部安全且预算内到达。机制证据不作学习/开发性能声明。

工作根为`/pub/data/caohy/work/camra_navigation/nav064_prior_v2_repair_20261117_v1/`；tiger11、原tmux会话、绝对Python不变；224份开工及60份收尾源码配置、33项v2身份和旧checkpoint散列保全。当前源码171父门禁/196项v1兼容/219项v2累计门禁、v2实际两更新PPO（36完整回合/11581样本）与独立审计、32回归及新增提案绑定全部通过，见`reports/nav064_avoidance_prior_v2_release_v1.json`。这些是包含关系和有限机制证据，不是正式训练或开发通过。详见[回修记录](nav064_avoidance_prior_v2_20261117.md)。

本次不转移旧HELD额度，不自动新增正式训练。拟交付独立未批准v2提案6×300、上限14580000样本及108开发回合；首50计入300，保留原探索/审计/关闭档。确认具体额度后才另冻登记/pin并启动；NAV-065及下游不放行。下文旧记录完整保留。

## 2026-11-17T14:35:18.564839+08:00 绕障探索版六组已启动，实际生产核验通过

训练在`tiger11.ios.ac.cn`的`nav064-r14-repair-20261115:avoidance-training`运行。启动提交`3f762b1729e790ba04f6711998ab2d757ea532af`，冻结pin为`d6bac4f626f31f437ae9cab4f02f02b117b47469b0498b59bc23c63d24df247f`。核验时六组更新数[13, 15, 13, 17, 15, 15]，累计354336样本，六进程正常且无HELD；18图、输入hash、绕障先验、完整回合及多epoch实际更新均通过核对。

首50更新原始审计与每个full种子安全L2到达前提尚待完成；通过才继续到300及108开发验收。此为启动证据，不能解释为学习或安全效果已达标。证据见`reports/nav064_avoidance_prior_started_v1.json`及独立launch工作根；NAV-064仍false，NAV-065+未执行。下方均为历史时点。

## 2026-11-17T14:31:58.671470+08:00 绕障探索版额度已批准，准备正式启动

用户已批准两方法×三种子各300完整回合更新（最多14580000样本），最后checkpoint的108回合development。独立登记`config/training_loop_prereg_avoidance_prior_v1.json`及SHA256 pin已冻结，原未批准proposal保持原样。前50计入300，先原始审计，再要求三个full种子各至少一次全程安全L2到达；否则HELD，保留现场且不自动重试/补额。

本次tiger11/tmux现场预检通过：196门禁、32回归、59份源码/配置快照及33项固定v2身份均匹配，六个新run与协调器/审计/开发输出无冲突。证据根为`/pub/data/caohy/work/camra_navigation/nav064_avoidance_prior_launch_20261117_v1/`。启动后以实际launch/start/pids与原始日志为准；当前NAV-064=false，全部训练/开发验收通过后才推进NAV-065。

## 2026-11-17 本次：完整回合版负结果结案，绕障探索候选通过196项数值/生产门禁

旧`nav064_episode_feedback_v1`六组各300更新均已结束，原始审计PASS，108开发回合FAIL_DEVELOPMENT。主预算下uncalibrated安全可行SR中位2/3、能耗/预算0.819487、危险步0；full分别1/2、0.869677、危险步600。full的L2训练5400回合到达1234次、全程安全到达0次；总体50%不能作为绕障或原关闭档通过依据。旧run、原始chunk、各checkpoint和失败全部保留，保全副本/hash由本次release报告索引。

NAV-064内已实现`clearance-tangent-v1`策略均值先验，仅使用当前obs-v7的目标/航向、障碍及墙净空，网络残差可抵消；采样、PPO、开发与原始审计使用同一分布，没有动作外接覆盖、oracle路线入训练或目标变化。当前门禁`reports/gate_avoidance_prior_v1_attempt1.json`为196/196 PASS，含源码稳定后重跑的171项父门禁及真实两更新机械生产采集、旧概率/critic回放、三目标审计和元数据反例。机械夹具不作为性能/选择证据；最终回归及源码/资源一致性以`reports/nav064_avoidance_prior_release_v1.json`为准。

具体推演、接口、局限与复算入口见[绕障探索记录](nav064_avoidance_prior_20261117.md)。独立工作根`/pub/data/caohy/work/camra_navigation/nav064_avoidance_prior_repair_20261117_v1/`，主机tiger11，tmux`nav064-r14-repair-20261115`，Python为`/pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python`。沿用已批准的同机执行例外；58份旧源码/配置及33项固定v2身份开工核验，原锁不重算。

提案`config/training_loop_avoidance_prior_proposal_v1.json`保持`training_approved=false`，无pin；拟新增两方法×三种子各300更新，上限14580000样本，最后checkpoint的108回合development。前50计入300，审计后要求三个full种子各至少一次安全L2到达，否则HELD，不自动重试/补额。原正式性能与稳定性门槛均保留；旧额度已耗尽，新增正式次数待用户确认后另冻登记。NAV-064=false、51/67历史完成，NAV-065及下游未执行。下方各状态为历史时点。

## 2026-11-17T07:21:42.759685+08:00 完整回合反馈版已启动，50更新审计通过

用户批准的两方法×三种子各300次更新正在`tiger11.ios.ac.cn`的`nav064-r14-repair-20261115:episode-training`运行。启动提交`55fbd6bff88041b71dbe3135c18c9ea63840c3a1`，独立冻结pin `f48db2cd9777854acb6963db1075241ba8b4bcbe27c10c5a29a8820263415cb1`。启动核验时六组完成[76, 75, 79, 78, 77, 80]次更新、实采2038253条，实际train18图、输入hash、obs-v7/61、完整当前策略回合及多epoch更新均核对，六进程存活，无HELD。

首50更新原始审计为PASS：六组5400完整回合、1279589样本，216段轨迹独立重算三目标，元数据不一致0。审计后训练已继续；此结果仅证明启动与训练机制/数据符合登记，尚不表示学习或导航安全达标。证据见`reports/nav064_episode_feedback_started_v1.json`，首轮原始审计位于`/pub/data/caohy/work/camra_navigation/nav064_episode_feedback_v1_audit_u50/report.json`。

协调器目录`/pub/data/caohy/work/camra_navigation/nav064_episode_feedback_v1/`；日志`/pub/data/caohy/work/camra_navigation/nav064_episode_feedback_launch_20261117_v1/coordinator.log`。六组各300后自动全量审计，再以最后checkpoint执行108回合development。接手先查start/pids/HELD/FINISHED和实际日志，不重复启动。无额外性能smoke、自动重试、推送或下游执行；NAV-064仍false，按原开发关闭档通过后才推进NAV-065。下方“尚未启动/待确认”均为历史时点。

## 2026-11-17 用户批准完整回合反馈版，冻结登记通过启动核验

用户已确认新增两方法×三种子各300次完整回合更新（最多14580000样本）及最后checkpoint的108回合development。独立冻结件为`config/training_loop_prereg_episode_feedback_v1.json`，pin `f48db2cd9777854acb6963db1075241ba8b4bcbe27c10c5a29a8820263415cb1`。原proposal和下方“待额度确认”保留为历史。训练前放行为`reports/nav064_episode_feedback_training_release_v1.json`；171门禁/32回归来源仍匹配，57份源码/配置快照和33个固定v2资源核验通过。

本轮协调器入口为`scripts/run_feedback_round.py --prereg training_loop_prereg_episode_feedback_v1.json`，在tiger11、tmux `nav064-r14-repair-20261115:episode-training`中执行，cwd为导航目录，解释器`/pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python`。run前缀`nav064_episode_feedback_v1`，工作根`/pub/data/caohy/work/camra_navigation/`，启动证据及日志在`nav064_episode_feedback_launch_20261117_v1/`。

先完成必要本地提交再启动；六组均到50更新后暂停审计，通过继续到300，全量审计后只评估最后checkpoint的108回合development。无自动重试、quota填充、额外性能smoke或下游执行。NAV-064仍false，需训练/开发/机制/数据审计全部满足后才关闭。恢复先核对协调器start/pids/HELD/FINISHED与实际进程，不重复启动旧run；原同机执行例外继续有效，不重复询问SSH。

## 2026-11-17 完整回合反馈候选通过生产数值门禁，正式训练尚未启动

净空版6×600已完整结束：原始审计PASS，108回合开发FAIL_DEVELOPMENT。主预算下两方法可行SR中位均1/3，但full危险步中位780，未通过安全关闭档。full全部600更新的L2训练到达1769次，安全/可行到达均0；不能据到达率推进下游。

NAV-064内完成独立候选：`complete_episodes_v1`采集全部18图每次各一个完整当前策略回合，三套估计器λ=.999；保留全部提前失败，不补样本、不跨更新带旧策略前缀。min共同停止、不可逆安全历史、目标/折扣、obs-v7/61、fixed v2、地图/预算/评分不变。实际生产采集/更新/原始审计门禁`reports/gate_episode_feedback_v1_attempt2.json`为171/171 PASS；36个机械回合/8439样本与180个minibatch均核对。受控六图×三critic对照18/18方向正确，只是干预诊断，尚不能证明探索或训练成功。

详细诊断、变更、完整证据及恢复入口见[完整回合反馈记录](nav064_episode_feedback_20261117.md)，最终回归与保全见`reports/nav064_episode_feedback_release_v1.json`。工作根`/pub/data/caohy/work/camra_navigation/nav064_detour_diagnosis_20261117_v1/`保留116份旧源码/配置/报告快照，继续已批准的tiger11同机/tmux例外，无需再问SSH例外。

提案`config/training_loop_episode_feedback_proposal_v1.json`保持`training_approved=false`、无pin；请求新增两方法×三种子各300完整回合更新（最多14580000样本）及最后checkpoint的108回合开发。前50计入300，审计50/300，无额外性能smoke、自动重试或下游执行。新增正式次数须确认后再冻结/启动；当前51/67历史完成，NAV-064=false，NAV-065及下游未执行。下方“正在训练”等均为历史时点，禁止重复启动。

## 2026-11-16T23:02:47.488078+08:00 用户批准后的净空版本启动

两方法×三种子各600更新及108开发回合已明确批准。独立冻结件为config/training_loop_prereg_clearance_feedback_v1.json，pin 776b86bfbeb874c5024645653859afb5d121d7b0520b6367a8b2d7dedceb3bb5；启动提交6e86cd0787e590bb8d6dd8ae894263e3d25cee8f。旧proposal和下方“待批准”表述均保留为历史。

六组已在tiger11的tmux nav064-r14-repair-20261115:clearance-training启动，当前各完成[1, 1, 1, 1, 1, 1]次更新。实际61维obs/next_obs首chunk独立净空oracle、train输入hash与初始更新有限性均通过；本条不表示训练效果已通过。证据见reports/nav064_clearance_feedback_started_v1.json。

协调器工作目录为/pub/data/caohy/work/camra_navigation/nav064_clearance_feedback_v1，日志为/pub/data/caohy/work/camra_navigation/nav064_clearance_feedback_launch_20261116_v1/coordinator.log。六组均达到100更新后暂停并审计，通过继续到600，再全量审计和最后checkpoint的108回合development。没有自动重试或下游执行，NAV-064仍false。断线后先核对已有pids/start/HELD/FINISHED与原日志，不复用run重复启动。

## 最新状态：2026-11-16晚间 净空接口通过门禁，NAV-064仍未通过训练验收

time-feedback-v1已完整结束：6组×600更新、全量审计PASS，开发FAIL_DEVELOPMENT。uncalibrated可行SR中位1/6、能耗/预算0.863326；full为1/2、0.890408，但危险步中位228。不能按下方旧“正在训练”重启，也不能据到达率放行。

本次按用户授权完成安全净空单一候选obs-v7：原41字段保留，加16障碍/4墙面有符号净空；full/uncal各用实际冻结几何半径，总61维。训练、开发、三网络、审计、协调器和严格启动核验已接通。最终门禁`reports/gate_clearance_feedback_v1_attempt4.json`为153/153，32套回归通过；真实边界一步动作及实际PPO受控更新方向正确，不能据此声称导航改善。固定v2、RC-PPO目标与gamma/lambda、地图、预算、评分不变。

最新train/development诊断和源码/资源保全分别见`reports/nav064_clearance_feedback_diagnosis_v1.json`、`reports/nav064_clearance_feedback_release_v1.json`；实施顺序、验证边界、复算命令与下一入口见[安全净空回修记录](nav064_clearance_feedback_20261116.md)。工作根`/pub/data/caohy/work/camra_navigation/nav064_clearance_feedback_repair_20261116_v1/`，继续既有tiger11同机/tmux授权。当前51/67历史完成，NAV-064=false，NAV-065及下游未执行。

独立提案`config/training_loop_clearance_feedback_proposal_v1.json`请求新增6组×600更新及108开发回合，尚未批准、冻结或启动。原time-feedback额度已用尽，按计划的正式次数约束待确认新额度；不因数值PASS自动训练或推进下游。下方各“最新”标题均为历史时点；旧feedback-v1的uncal SR中位数写成0属于文字误记，应以原开发报告实际2/3为准，旧文保留供追溯。

## 最新状态：2026-11-16 时间接口修复完成数值门禁，训练提案待新增额度

feedback-v1六组各600更新已结束，开发结论`FAIL_DEVELOPMENT`：uncalibrated安全可行SR中位数0，full为1/6，但full危险步中位数654。下方“正在训练”等保留为历史时点，不能据此重启旧run。

NAV-064内已增加显式obs-v6（剩余时间除45秒），修复前15秒时间编码饱和；旧obs-v2至v5按原编码复现。训练、开发构造器、审计和协调器均支持新版本与独立run命名空间。最新数值门禁为`reports/gate_time_feedback_v1_attempt3.json`，120/120通过；源码/回归/资源保全汇总以`reports/nav064_time_feedback_release_v1.json`为准。

旧轨迹诊断证实部分首次越界被距离项遮蔽、安全历史饱和形成平台；λ离线变化方向不一致，仍保留0.95。详细证据、局限和下一入口见[时间修复与诊断](nav064_time_feedback_20261116.md)。仅时间候选提案为`config/training_loop_time_feedback_proposal_v1.json`，申请新增六组×600更新，`training_approved=false`，尚未开训。原六组额度已耗尽，需确认新增次数后另建冻结预登记；没有额外性能smoke。NAV-064仍false，不能推进NAV-065及后续。


## 最新启动：2026-11-16 feedback-v1 六组训练运行中

用户已批准新六个 fresh-init run ×600更新（共14709600样本），并完成独立冻结预登记 `config/training_loop_prereg_feedback_v1.json`，SHA256 `d14b301a2a26d3edfba32e6fe73ec8cb489a8cacad859f81ddcd68943b657459`。原proposal及失败记录保留，下文“待额度确认”描述此前时点。

2026-11-16 11:41:49，训练从本地提交 `41a93723dfd86ddc7ab8260a063d10aa6a93fa82` 在 `tiger11.ios.ac.cn` 的 `nav064-r14-repair-20261115:feedback-training` 启动；解释器为 `/pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python`。入口为 `scripts/run_feedback_round.py --prereg training_loop_prereg_feedback_v1.json`，cwd为本模块。run为 `nav064_feedback_v1_{uncalibrated,full}_s{0,1,2}`，输出根 `/pub/data/caohy/work/camra_navigation/`；协调器目录 `nav064_feedback_v1/`，启动日志 `nav064_feedback_launch_20261116_v1/coordinator.log`。

启动核验时六组各已完成6次更新，共147096个样本；实际使用train18图、obs-v5/rcppo-feedback-v4，每次4086样本、10 epoch×512、80梯度步。冻结pin与实际加载输入散列一致，已观察损失/梯度有限、策略梯度非零、六进程存活且无HELD。证据见 `reports/nav064_feedback_started_v1.json` 及其外部原始记录。这只证明启动和初始更新，尚不表示学习效果或安全可行到达达标。

协调器在六组均达100次后暂停并严格审计首100次（计入600），通过才继续；600次后全量审计并评估最后checkpoint的development网格。失败保留现场，无自动重试或追加轮次，不自动进入NAV-065。断线后先查 `nav064_feedback_v1/{pids,start,HELD,FINISHED}.json`、进程和日志，禁止重启同run。NAV-064仍false，下游未放行。

## 最新状态：2026-11-16 反馈回修通过数值门禁，新增训练待额度确认

r14a2六组各600更新已结束，实际开发结论为`FAIL_DEVELOPMENT`：uncalibrated安全可行SR中位数0，full为1/6，但full危险步中位数114，未通过原关闭档。以下“正在训练”保留为此前时点记录。

NAV-064内已完成新版有限任务终端、外部窗口bootstrap区分、全分级安全样本学习及安全延续值边界。当前源码绑定门禁`reports/gate_feedback_v1_attempt4.json`为97/97 PASS；73548条已有train样本重算通过，旧目标可复现。原始失败、旧契约和源码快照保留。联合gamma=1的依据已重做有限时域推导，未再以机械功不折扣替代该推导。

32套回归通过；受影响前置NAV-062/063恢复true。收尾证据为`reports/nav064_feedback_repair_release_v1.json`，只通过数值/回归验收，尚无新版训练效果。

详细结果、验收边界、资源提案及下一入口见[反馈回修记录](nav064_feedback_repair_20261116.md)。新增六run×600更新提案是`config/training_loop_feedback_proposal_v1.json`；`training_approved=false`，尚未启动，不把数值正确等同于导航训练通过。NAV-064仍false，下游065–067未执行；任务状态与回归汇总以主计划和本次release报告为准。

## 最新启动证据：r14a2 正在训练

2026-11-15 23:47:33，六个新 run 已从本地提交 `4b793d1d59e97d3f1c7db9f7dfc2252279852109` 在 tiger11/tmux 的 `nav064-r14-repair-20261115:r14a2-training` 启动。每组已完成至少两个正式更新；独立检查六组前两次更新的共49032个原始样本，确认18×227窗口、10 epoch×512、gamma_h由0.999升至0.9990033、gamma_joint=1及原始回合身份/写入hash匹配。具体PID、进度与路径见 `reports/nav064_r14a2_started_v1.json`。

协调器会在首100更新后严格审计，在各600更新结束后全量审计并评估最后checkpoint的development表现。当前训练与性能验收均未完成，NAV-064仍false，NAV-065及后续未启动。旧attempt与失败全保留。下方“等待提交后开训”是本次启动前的历史放行状态。


## 最新执行状态：r14a2 已放行，等待本地提交后开训

用户已明确批准一次新增六个 fresh-init run ×600 更新及必要本地提交。旧六个有实现偏差的 attempt 已按 PID 核验后结束，368 个完整更新、未完成窗口、原日志、原始数组和 u50 checkpoint 均保留；旧报告和冻结预登记不改写。保存与终止证据为 `reports/nav064_r14a2_preparation.json`。

修复已接入正式源码：reach 使用参考实际 gamma_2 双倍调度，半程达到 0.99999；reset 前保存回合标量身份；训练、评估、审计从冻结 registry 读取本次 run ID。新门禁 65/65、32 套回归全部通过，放行报告为 `reports/nav064_r14a2_training_release_v1.json`。活动源码、消费链和新门禁已绑定 `config/training_loop_prereg_r14_repair_v2.json`（r14.2.0，SHA256 `87e814ae0de8aee5276627232e6eff4c8d77b339a9e8568e16e7e337aa19eff2`），放行快照见报告。

新 run 为 `nav064_r14a2_{uncalibrated,full}_s{0,1,2}`，每 run 600 更新、每更新18×227=4086样本；前100计入600。入口 `scripts/run_round14a2.py` 在 tiger11/tmux 核验预登记与重名后启动六组，自动执行第100更新前缀审计、600更新全量审计及最后 checkpoint 的 development 评估。失败会保留现场并停下；不会自动推进 NAV-065 或追加新轮次。输出根为 `/pub/data/caohy/work/camra_navigation/`，本轮协调日志目录为 `nav064_r14a2/`。

目前只通过训练前验收，NAV-064 仍 false。两方法三种子中位数可行 SR≥0.10、能耗/预算≤1、full危险步中位数为0及机制/稳定性/数据审计需全部满足才可关闭。corr 只作诊断，不能替代门禁或性能。固定v2、旧校准、预算选择集与final封存、只读参考和论文边界保持。

下列“暂停”“待授权”“尚未替换”等文字均保留为此前时点的历史记录，当前资源授权已解除。


## 当前恢复入口：2026-11-15 NAV-064 r14

**23:03 起的最新状态：六个训练进程已暂停，NAV-064 仍 false。** 生产入口漏接冻结契约要求的 reach 折扣退火，实际一直使用 0.999；真实样本重算确认会改变 reach 目标。另有终止后写摘要时引用下一回合身份/预算的问题，逐步原始数组保留正确。原放行报告未检查退火消费者，不能继续作为完整链路的放行证据；旧报告不改写。当前已完成更新数为 uncalibrated 65/62/64、full 59/59/59，共 368，六个进程均 SIGSTOP，u50 存档及原日志保留。

修复仅在隔离候选中完成：接入 RC-PPO 实际 gamma_2 调度（半程达到 0.99999），并在 reset 前捕获回合标量身份。专项 14 项、独立数值门禁 37 项和 8 套针对性回归通过；最新复现脚本为 `docs/patches/verify_train_formal_r14_gamma_meta_v4.py`，匹配补丁为 `docs/patches/train_formal_r14_gamma_meta_v3.patch`。活动训练源码及原冻结预登记尚未替换；不得直接恢复旧进程或复用原 run ID。修复证据与六个新 run 的具体重训提案见[候选复核与重训提案](nav064_r14_gamma_meta_review_20261115.md)。追加正式 attempt/资源尚待用户确认；不是 SSH 同机例外待决，也不表示训练达标。以下启动信息均保留为暂停前历史。

2026-11-15 22:50 实际启动更新：训练前32套回归及37项独立数值/生产调用门禁全部通过，放行证据见 `reports/nav064_r14_training_release_v1.json`。六个正式run（`nav064_r14_{uncalibrated,full}_s{0,1,2}`）已在 tiger11 的 `nav064-r14-repair-20261115:round14` 启动，输出位于 `/pub/data/caohy/work/camra_navigation/`。每run按原额度执行600更新、18×227窗口，前100更新计入600；最后checkpoint用于既定开发网格。训练、开发指标与审计尚未完成，NAV-064仍false，下游未放行。以下“尚未启动”均为此前时点事实。

主计划 1.3.1 当前为 **51/67 项历史记录完成，NAV-064=false**。用户同意先回修学习链路、通过独立数值和实际更新门禁，再在原额度内训练；开发验收通过后才进入 NAV-065 及下游。具体顺序、接口要求、反例和资源边界见[当前阶段恢复安排](nav064_r14_recovery_plan_20261115.md)，执行主体记录追加在主计划 NAV-064 下。

2026-11-15 续跑更新：用户已明确允许当前 tiger11 会话直接用 tmux 执行，原 SSH 失败作为已授权的同机例外保留，不再是当前阻塞。会话为 `nav064-r14-repair-20261115`；v2 29 个冻结文件、匹配校准输入与资产核验通过，修正前源码保存于 `/pub/data/caohy/work/camra_navigation/nav064_r14_repair_20261115_a2/source_before/`。新版 obs-v5 / rcppo-r14-v3 已完成回修，独立数值与生产消费者门禁见 `reports/gate_r14_repair_v1_attempt3.json`（37/37 PASS），旧 FAIL 与前两次 attempt 保留。新训练预登记为 `config/training_loop_prereg_r14_repair_v1.json`，SHA256 `30033056d6ca2b27d636998ee9d85b573e791e53ceee4a78c3e127299a206c0a`。回归复验正在收尾，训练尚未启动；NAV-064 仍 false。

以下段落保留为本日取得同机授权前的历史状态：

本次只完成静态核查与恢复安排：当前终端为 `tiger11.ios.ac.cn`，但规定的 SSH 身份预检返回 Permission denied（退出码255），因此实现、数值重验和训练仍未启动。需先恢复规定登录检查，或取得同机直接 tmux 执行的明确例外。原 r14 预登记、两次失败报告、已有修改和历史记录保留；旧 true/PASS 不代替当前新版放行。

下面的“当前交接：2026-11-12”及其48/67、下一项061等表述保留为当时的历史快照。

## 当前交接：2026-11-12计划修订

当前[实施计划](../../../doc/plan/20261109_camra_navigation_implementation.json)为 **1.3.1，48/67项记录完成**，下一项 **NAV-061**。本次只修订现有任务与说明，未运行服务器、实现、训练或新评估；保留全部任务编号、数组顺序、历史状态与工作记录。

**已存在的实现不等于已经达标。** 名义环境、三网络训练、预算工具及物理接入已建立，但旧三轮开发集成功率为0，`reports/gate_G4.json`为`FAIL_WITH_STABILITY`。NAV-039经用户接受负结果结案；旧报告和冻结候选继续保留，不以新计划改写旧结论。

| 历史产物 | 当前可证明范围 | 本轮责任任务 |
| --- | --- | --- |
| 三轮名义训练、G3/G4与旧部署候选 | 已实现并运行过；共同停止、三估计器、时序及部署价值仍需回修/重验 | NAV-061–065 |
| `reports/budget_selection.json`的90回合 | 单种子经验标签扫描且全无解，不是双critic价值搜索 | NAV-066 |
| `reports/nav042_online_loop.json`的6回合 | 10/50/200 Hz接线pilot；clean随机项和子步事件证据不足以证明新要求 | NAV-067 |
| 原v2校准、旧G2A | 固定控制器和原采样条件的历史证据，不为未来学习策略自动背书 | NAV-043/048 |

恢复顺序为：NAV-061冻结原理、回修、数据和资源协议 → NAV-062对齐时序/预算/终止/动作接口 → NAV-063修正共同停止与三估计器并查生产调用链 → NAV-064有限名义训练 → NAV-065新版actor/同目标双critic及G4 → NAV-066独立价值预算搜索 → NAV-067新版候选clean子步接线 → NAV-043–054正式复查、评估与交付。NAV-067依赖NAV-066，不复用旧候选跳过重验。

训练只用train/development调参及选checkpoint，登记旧选择集暴露；独立选择与`final_*`不回流训练。90回合只覆盖预先指定的单种子/方法；新的独立清单、重复性、pilot或资源额度按计划取得必要授权。旧失败、全部attempt与证据保留，回修主体记录追加到发现问题的当前任务。

固定v2整套执行/校准资源，执行主机固定tiger11（192.168.20.11）+tmux。下文服务器路径、解释器和会话是历史线索，恢复前现场核验，不因能看到共享仓库而推定执行主机；本次没有重新核验其可用性。v2、旧校准、RC-PPO/HiRA和论文不在修改范围。

## 历史快照：2026-11-10（NAV-001–010完成时）

以下保留当时的交接内容；其中“未开始”“下一项NAV-055”等仅描述该历史时点，不是当前任务入口。

更新时间：2026-11-10T10:10:00+08:00。执行主机 tiger11（192.168.20.11）。

### 1. 已完成与未开始

| 阶段 | 任务 | 状态 |
|---|---|---|
| 背景学习 | NAV-001–006 | 完成，产出 onboarding.md 与 source_reference_map.md |
| 服务器与资源 | NAV-007–010 | 完成，产出 runtime_environment.json、runtime_paths.json、v2_resources.json、包骨架 |
| 契约与名义环境 | NAV-012–023（含 055–058） | **未开始**，等 G0 |
| 后续 | NAV-011 及之后 | 未开始 |

**本阶段没有任何仿真、训练、采集或依赖安装以外的动作**；唯一对环境的写入是 editable 安装了本仓库的 `camra-navigation` 包。

### 2. 目录清单

**仓库内（现有 + 本次新建）**

| 路径 | 性质 | 说明 |
|---|---|---|
| `CaMRA/navigation/README.md` | 现有 | 模块说明 |
| `CaMRA/navigation/config/runtime_paths.json` | NAV-008 新建 | 路径、解释器裁定、框架决定、依赖、隔离方案 |
| `CaMRA/navigation/docs/onboarding.md` | NAV-001/002/004/006 | 基线、方法、v2 契约、六阶段矩阵 |
| `CaMRA/navigation/docs/source_reference_map.md` | NAV-003/005 | RC-PPO 调用链、HiRA 分层结构 |
| `CaMRA/navigation/docs/runtime_environment.json` | NAV-007/008 | 主机、tmux、依赖、冻结环境身份 |
| `CaMRA/navigation/docs/implementation_handoff.md` | NAV-010（本文件） | 交接 |
| `CaMRA/navigation/reports/v2_resources.json` | NAV-009 | v2 资源核验报告 |
| `CaMRA/navigation/pyproject.toml` | NAV-010 新建 | 包定义 |
| `CaMRA/navigation/src/camra_navigation/` | NAV-010 新建 | `__init__.py`、`runs.py`（运行身份规则） |
| `CaMRA/navigation/tests/`、`scripts/` | NAV-010 新建 | 空目录，等后续任务填充 |

**仓库外（新建，与旧产物隔离）**

| 路径 | 说明 |
|---|---|
| `/pub/data/caohy/work/camra_navigation` | 工作根（代码快照、中间产物） |
| `/pub/data/caohy/logs/camra_navigation` | 运行输出根 |

两者均不使用、不写入 `logs/camra`（v2 校准）或 `logs/camra_hira`（HiRA 实验）。

### 3. 运行身份与冲突拒绝

`camra_navigation.runs` 已强制三条：

1. run id 必须匹配 `^[a-z0-9][a-z0-9._-]{2,79}$`；
2. 目标目录已存在时**默认拒绝**；即便 `allow_existing=True`，非空目录也拒绝；
3. `run_manifest.json` 不可覆盖，写入 hostname、started_at、cwd、python 与 controller_id。

实测三个反例均被拒绝（重名、非法 id、非空目录），正例可创建并写出 manifest。

### 4. 环境

- 解释器：`/pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python`（Python 3.11.15）
- isaaclab 0.47.2 / isaacsim 5.1.0.0 / rsl-rl-lib 3.0.1 / torch 2.7.0+cu128 / numpy 1.26.0 / onnxruntime 1.31.1
- 关键偏差：torch 为 cu128（冻结记录 cu126）、Python 3.11.15（记录 3.11.18）——已登记，未修正
- 上层框架：**PyTorch**（用户 2026-11-10 批准）
- 包：`camra-navigation 0.1.0`，editable 指向 `src/camra_navigation`

### 5. 下一步入口

**下一个可执行任务是 NAV-055（速度域冲突裁定）**，不是 NAV-011。

- NAV-055 需要：依据原始 applied 命令抽样 + 数据锁，判定有效命令域是 `1.0/0.4/1.0` 还是 `2.0/1.0/2.0`。
- 已确认事实：protocol、v2 contract、power_model.domain 三方一致为 `1.0/0.4/1.0`；仅 `margins_v1.applicability` 的**文本**为 `2/1/2`。
- 该裁定未完成前，G0（NAV-011）不得放行。

### 6. 未决点

| 项 | 负责 | 阻塞 |
|---|---|---|
| 速度域裁定 | NAV-055 | G0 |
| 姿态足迹处理 | NAV-057 | G1 |
| z_ref 物理定义 | NAV-056 | G1 |
| 预算公平性协议 | NAV-058 / NAV-021 | 正式训练 |
| JAX 对照环境与依赖授权 | NAV-033 | G3 |
| 地图净宽与 30 s 可达 | NAV-017 | G1 |

### 7. 恢复方式

```bash
ssh 192.168.20.11 -t tmux attach -t camra-nav-v2-20261110-bootstrap
set -a; . /pub/data/caohy/v2_navigation.env; set +a
cd /home/caohy/repositories/Paper
```

断线后先 `tmux ls` 与查进程，不重复启动同一 run。

## NAV-064时间观测版训练启动（2026-11-16）

用户已批准time-feedback提案：两方法×三种子，fresh-init各600更新，首100计入600，共14709600样本；最后checkpoint执行108回合development。独立冻结件为`config/training_loop_prereg_time_feedback_v1.json`，pin `b8742ca9c50629e0cf350ea5d6328339c86f70239b32d73444865ad5bb2a3216`，启动源码提交`c850b69`。原提案、旧run与所有失败保留。

六组于17:33在tiger11、tmux `nav064-r14-repair-20261115:time-feedback-training`启动。解释器`/pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python`；cwd为导航目录。协调器目录`/pub/data/caohy/work/camra_navigation/nav064_time_feedback_v1/`，启动日志及证据在`/pub/data/caohy/work/camra_navigation/nav064_time_feedback_launch_20261116_v1/`。启动报告`reports/nav064_time_feedback_started_v1.json`记录六组PID、pin、实际输入hash与最初更新；六份首chunk已验证45秒时间编码。

协调器自动在100更新后暂停作严格审计，通过后继续至600，再全量审计及开发判定；出现HELD须先查现场，不能重复启动或加跑。当前仅证明启动及初始机制，NAV-064仍false；原关闭档不变，NAV-065及后续未执行。恢复时先读根AGENTS和计划最新记录，再检查上述会话、进程、run及HELD/FINISHED标记。

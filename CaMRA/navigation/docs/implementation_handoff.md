# 上层导航实施交接

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

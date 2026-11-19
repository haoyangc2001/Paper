# 上层导航实施交接

## 2026-11-19 NAV-043 L2自选预算后跟踪差距仍在，原L3继续

在上一结果提交9d206a3后，主agent只读检查L2两条保存轨迹，时间/tick、10Hz命令与200Hz持有、W决定pin及名义初态/预算、净航向解缠和来源hash基本核对通过。完整方法平均前向0.398指令对0.324m/s实际，净转向129.8°对38.8°；名义42.6秒到达只有2.4秒余量，物理45秒仍距目标2.75m且仅耗所选预算75.41%。这补强模型跟踪差距仍在的描述，不定位唯一物理原因，也不以单次比较估计预算的因果作用。代码/快照/日志在.worklog-testing/NAV-043/exploration-v1/analyze_L2_tracking.py及L2-tracking-diagnostic.json/.log，详见[说明](nav043_L2_budget_tracking_review_20261119.md)。

原协调器38516/L3应用338561及%601/59048、%612/317768继续；本次新增应用/物理或名义回合/W/拟合/训练0，不重复审阅入口或试验。最终事件和适用性复核尚待原链完成，NAV-043=false、55/67及G5缺口保持。下一方案先针对实际模型差距和45秒适用性评审，不默认增加预算/critic训练；任何新校准或参数改动另作范围/额度登记。本地提交，不推送，下方为保留历史。

## 2026-11-19 NAV-043 L2完整方法超时但未超预算，原分配进入L3

原协调器38516已正常结束L2应用173883，EXIT累计4180.373219秒、L2显式物理步18400，随后原协议既定L3应用338561在同一根目录/GPU6运行，无重试或新分配。原%601/PID59048保存物理事件复核及%612/PID317768适用性复核的等待链保持，不另外手动重跑入口。

完整方法L2原实际初态W规则选择k1.3/640.371590J；实际45秒timeout、安全历史0、实际482.893260J、内部548.792469J，分别只用所选预算75.41%/85.70%，终态仍距目标2.750487m。同初态/预算名义42.6秒goal_reached/516.878633J，只有2.4秒名义时限余量。此次失败是到达时限，不能解释为预算太紧，也不能仅凭单回合证明唯一物理根因。无校准L2仍45秒超时且超预算2.11%，原失败全部保留。

主agent在tiger11/tmux %614只读保存L1/L2四结果及22份decision/pin/result/nominal/应用FINISHED/EXIT/计数来源hash，调用save_L2_progress.py后再次核对来源一致；产物L2-completed-progress.json/.log。它只整理已保存在线结果，没有新仿真、模型推理或完整物理事件重放，不是最终审阅或科学通过。仍55/67、NAV-043=false、G5未通过，原45秒/位置裕度适用性缺口保持。L3及原两步审阅结束后再做全量结论，固定v2、模型、科学门槛和alpha/delta不改；本地提交，不推送。下方为保留历史。

## 2026-11-19 NAV-043只读适用性复核自动接续原审阅器

上一轮为已核实运行等待，最近准备提交6fa0cd3已核实。原协调器38516/L2应用173883仍执行同一有限分配，不重启。原%601/PID59048全物理事件审阅等待器继续；本次新增%612/PID317768的applicability_when_reviewed.py，仅在该原PID及/proc启动身份结束、完整审阅报告存在后，调用一次已验证audit_nav043_exploration_applicability.py。不再另行手动运行同一适用性入口。

主agent实际--check验证host/tmux/绝对解释器、原等待器存活及cmdline/启动身份、入口hash、输出和启动记录拒绝覆盖；legacy计划校验通过。证据applicability-followup-check.log、applicability-followup-plan.log、applicability-followup-start.json。当前applicability-followup.log只有WAITING_FOR_ORIGINAL_REVIEWER，最终报告尚未生成，不能当作已审阅。原审阅失败/缺报告或后续复核失败均保留错误并停下，没有自动重审、模型推理、仿真重试或新增额度。两步全部结束后主agent仍须读取实际报告、逐项复核、追加结论和本地提交；不推送。

本次只补齐保存结果收尾调用链，新增应用/物理或名义回合/W/训练/拟合0；原封存生产代码、模型、v2、alpha/delta、45秒任务、科学门槛和历史均不改。NAV-043=false、55/67，G5与原适用性缺口保持。下方为保留的上一时点。

## 2026-11-19 NAV-043 L2无校准已保存，适用性只读入口准备完成

原协调器38516/L2应用173883仍在；L2无校准45秒timeout、安全历史0、实际485.877977J/预算475.842709J（超支2.11%），内部436.096172J未超预算，距终点1.091476m。名义同初态及预算37.2秒到达/356.671819J。完整方法L2已封存决定并开始，尚无终态。上述新结果仍待原%601全物理事件复核，不重启分配或挪额度。

新增scripts/audit_nav043_exploration_applicability.py只读消费原全物理事件审阅报告及其hash索引，复用未改的analyze_trial生成原alpha/delta与5/10/20/30/45秒、指令域对照；不拟合、不改变旧表、不把缺失前缀补成45秒。%610主agent用两条L1保存轨迹完成前缀基本验证及生产缺报告拒绝检查，证据applicability-ready-v1.json和拒绝日志；正式新报告reports/nav043_exploration_applicability_v1.json尚未生成。全分配及原%601审阅结束后才运行该入口，原%601无需重启。

L1完整方法费用相对同命令基础模型高5.45%，在原alpha19.05%内；最大位置偏差10Hz为9.508248m/200Hz为9.510286m，原delta2.613866m不覆盖。它是实际轨迹与同命令理想参考的差，不是定位误差或离目标距离。到达成功与裕度适用性分列，NAV-043=false、55/67、G5未通过保持。本次只读准备新增应用/回合/W/训练/校准0，本地提交、不推送，无测试子agent。以下为保留的上一时点。

## 2026-11-19 NAV-043 L1完成，L2原分配继续运行

先查原协调器38516和L2应用173883，原L1应用38860已正常结束，勿因该旧PID缺失重启。L1完整方法41.905秒安全到达，实际449.096733J、预算527.505420J（利用85.14%），内部507.003978J；名义配对34.3秒到达/407.595315J。无校准仍为45秒超时、实际443.321331J对391.974928J预算超支13.10%，名义38.3秒到达/333.443941J。两方法是不同既定actor及预算规则配置，单配置n=1，不能把全部差异归因于单个裕度，也不估计最低预算效率。

主agent在%607只读重放首条9000物理状态后，在派生数值比较处失败；原脚本review_first_saved.py及first-saved-review.log保留。独立机械功求和与在线累计分别差约1e-12J，原审阅器却要求由不同求和源派生的数值逐位相等。现仅修只读审阅器，以原在线总量精确复现原业务评分，原audit_episode独立积分误差检查不变；不改预算、原结果、科学阈值或任何封存生产源码。

%608调用同一audit_nominal_trace函数验证两条名义共726个保存动作的全时刻观测/状态/预算/安全及终态，17381物理子步命令/时钟/能量记账通过；报告first-nominal-review.json保留原来源hash和首失败日志hash。完整物理事件重放仍等%601在全分配FINISHED/HELD后执行一次，当前不是全分配或G5通过。该复核新增应用/回合/W/训练0，原分配继续，仍55/67、NAV-043=false及原45秒/位置裕度缺口。后续以新应用实际PROCESS/EXIT为准；原输出根和额度不变，不重试、不调参、不推送。下方为保留的上一时点。

## 2026-11-19 NAV-043首结果已保存，继续原分配

当前原协调器PID38516和L1应用PID38860继续运行，不向%599前台Python输入shell命令，不新开分配。首无校准原始记录为：W选择391.974928J，名义38.3秒到达/333.443941J；物理45秒超时/安全历史0/443.321331J，超支51.346403J（13.10%），剩余目标距离3.392034m。完整方法已按实际初态选择527.505420J后执行中。本段只记录已保存进度，不能外推完整方法或整个分配结果，仍待统一200Hz事件/能量复核。

只读审阅器已补充保存名义动作的float32逐时刻重放和未完成应用计数；语法/封存16项来源/计划状态基础检查通过，证据review-ready.json、review-ready-v2.log及review-ready-plan-validation.log。首次静态检查错误地把dict.values当网络推理，已修正且失败日志review-ready.log原样保留；没有物理重试。%601仍等唯一输出根FINISHED/HELD后只审阅一次；如只读审阅报错，保留原错误修复审阅器，不重跑应用。生产封存源码/资源/规则不变，NAV-043=false、55/67、G5和45秒/位置裕度缺口保持。收尾须看实际报告、追加结论和本地提交，不推送。

## 2026-11-19 NAV-043唯一独立分配实际运行中

准备封存已提交04c4cb3，随后tiger11/tmux %599启动协调器38516、首L1应用38860，GPU6。实际warmup状态已完成5个W查询及pin，首无校准选择391.974928J/k1.15并开始导航；尚无完整回合结案结果，不能报成功率或费用。首批10份不可变启动证据hash及进程见`.worklog-testing/NAV-043/exploration-v1/launched.json`。

启动命令只显式传GPU，遗漏了旧启动器的EULA/缓存/LD/OMP环境项；在原进程确认既有EULA后继续，没有重启应用。可选渲染库libXt启动消息保留；实际clean配置/材质/质量/碰撞体位置读回已生成，最终有效性仍须看原始轨迹和200Hz事件/能量复算，不能由进程存活判断。

新增scripts/audit_nav043_exploration.py只读审阅器，%601中的review_when_finished.py等待原分配FINISHED/HELD后只审阅一次；不会推理模型、采样或重试。输出报告预定reports/nav043_exploration_review_v1.json，目前不存在。续行先核对原协调器/子进程及唯一输出，不重启、不改封存来源/规则或挪额度。仍55/67、NAV-043=false，原位置/45秒适用性缺口及G5未通过不变；收尾须读取实际报告、追加结果并提交。

## 2026-11-19 NAV-043委托择路后启动有限探索路线

用户已授权择路并继续，选择已展示的有限探索性自选预算验证。独立[协议](nav043_exploration_protocol_20261119.md)/config/nav043_exploration_v1.json封存3应用、6物理、6配对名义、30初始W、单GPU3小时；使用原两候选、原W规则和已暴露三张integration_pilot图。每病例实际warmup状态五档W决定先写出及pin再行动；名义配对从同一实际平面初态/速度出发，物理费用与内部成本分列。拒绝保留无回退；业务失败保留并执行剩余预定项，工程/证据失败保存停止，禁止重试或扩额。

主agent5组基本检查和生产context295保护身份核验通过，没有专用测试子agent；准备时无真实W/新回合。先本地提交，再在tiger11/tmux %599、GPU6运行scripts/run_nav043_exploration.py。唯一输出`/pub/data/caohy/work/camra_navigation/nav043_budget_exploration_v1`，先看OWNER/进程及FINISHED/HELD，不能重复启动。NAV-043=false、55/67及原裕度适用性缺口保持，本轮不签G5、不访问final、不训练、不改v2/alpha/delta、不推送。结果以实际保存输出及主agent复核追加；以下为历史时点。

## 2026-11-19 NAV-043补充已保存跟踪诊断

本轮只消费已有12条轨迹；完整方法三配置平均实际机身前向速度较指令低约16%–18%，净转向指令/实际分别为61.1°/−8.8°、124.8°/38.9°、122.9°/43.5°。这支持跟踪偏差长期累积的描述，不定位唯一物理根因；机身wz与Euler航向、几何向量记账与因果贡献均分开处理。来源及逐项数值见[补充审阅](nav043_saved_tracking_review_20261119.md)与[报告](../reports/nav043_saved_tracking_diagnostic_v1.json)，代码/原始日志/主agent基本核验位于.worklog-testing/NAV-043/tracking-v1/。

60份原始输入、295项保护身份、净转向/保存参考、追加记录和保护路径基本核验通过。没有新应用/物理或名义回合/W查询/拟合/训练/final访问；固定v2、原α/δ、候选及评分不变。仍55/67、NAV-043=false、G5未通过，下一步等待上一轮两路线的明确范围与独立额度决定，不重启旧协调器或默认再训critic。本次修改按授权本地提交，不推送、不安排测试子agent；以下旧记录保留。

## 2026-11-19 NAV-043保存轨迹适用性与额度复审

主agent按legacy完成12条原/重复保存轨迹的命令参考、200Hz机械功和5/10/20/30/45秒前缀复算；70个来源hash条目、295项保护身份及基本验证通过。完整方法三个配置终态费用误差1.90%/4.81%/9.26%，均在α=19.05%内；共同预算仍全部超支。全程最大位置偏差9.318/9.269/9.998米是同命令理想参考与物理执行的距离，不是定位误差或终点距离。

固定δ=2.614米不覆盖已测累计偏差；旧闭环表虽在完整方法30秒前缀数值上覆盖，但仅针对vy=0的规则控制流且止于30秒，当前横向命令及45秒任务未验证。保留提前结束前缀缺失分母、不把重复池化为独立配置，不签NAV-043/G5。

旧诊断最新报告51+48=99条/回合，不构成完整去重账本或越权结论，也不能从旧not_started算可转用余量。新应用/回合/W/训练/final访问0；NAV-067工程通过保持，当前NAV-043=false、55/67。下一步需明确有限探索性预算验证或先制定新校准方案的科学范围，再实现并封存独立入口/额度；不能直接重启旧协调器。详见[本次结论与具体路线](nav043_saved_applicability_review_20261119.md)、[机器报告](../reports/nav043_saved_applicability_audit_v1.json)、`.worklog-testing/NAV-043/preparation-v1/saved-audit-verification.json`。修改本地提交、不推送，无测试子agent。

## 2026-11-19 NAV-043下一阶段范围准备，尚未新增执行

NAV-067已提交6125043，计划55/67；NAV-043保持false。仅准备[下一阶段范围](nav043_next_stage_scope_20261119.md)：先对账独立资源、建立新版入口和登记，再检查自选预算的真实物理效果及45秒裕度适用性。旧共同预算12轨迹已暴露，不替代新预算运行；旧规则pilot闭环表只到30秒，不外推、不反调冻结值。新增应用/回合/W查询/训练0，未获新试验批准，不签G5。源码/历史记录与失败保留，本地提交、不推送。

## 2026-11-19 NAV-067工程接入验收完成，下一项NAV-043未执行

NAV-067已逐条通过工程接入验收，计划1.4.1、55/67；6实际受控及原6加新6导航保存轨迹复核通过，每配置n=2只作描述。不是方法性能或G5通过，原科学FAIL、超时/超支和全部旧失败保留。下一项NAV-043尚未执行，当前恢复批准不含其新额度，不自动加跑。见[本次验收](nav067_engineering_acceptance_20261119.md)。下方运行中/false为历史时点。

## 2026-11-19 六项实际受控及保存读回重放通过，六导航重复运行中

已按一次恢复批准完成C01/attempt2、C02–C06，实际事件、active屏蔽和200Hz机械功重放复核通过；原失败不覆盖。C01–C05为人为写入后的实际读回，C06为原生接触，均非自然导航性能分母。六导航重复仍由原协调器PID2588869串行运行；只读审阅器等待RECOVERY_V2_FINISHED/HELD，全部结束后一次离线复核、不自动重试或标true。NAV-067=false、54/67、NAV-043未执行，完整验收和amendment待实际重复结果。见[进度与恢复说明](nav067_recovery_v2_progress_20261119.md)和[机器证据](../reports/nav067_remaining_recovery_v2_progress_20261119.json)。

用户同时询问未提交文件：已原样补交188小文件于3ca89c2；7大型逐步明细和1计划备份逐字节复制核验、原件保留，Git保存大小/hash/路径索引。历史空JSON失败保留，未执行旧脚本。当前运行尚未生成的终态证据后续审阅再提交，未推送。下方历史状态和科学FAIL保留。

## 2026-11-19 NAV-067恢复v2已获用户确认，开始一次执行

用户已确认隔离plane/grid修复及一次C01/attempt2恢复，累计应用上限10、GPU6串行累计4小时含旧118.444秒、60000环境子步含旧408预留；精确预留57830。APPROVED登记及pin另存`config/nav067_remaining_validation_recovery_v2.json`，原提案和失败保持。先核验生产context和保护身份、提交，再执行恢复入口；C01–C06全过才继续原六导航重复，任一新实际证据/运行/轨迹失败保存即停。NAV-067仍false、54/67，NAV-043未执行，无训练/W查询/final_*或推送。主agent按legacy验证，无测试子agent。并行只读核对工作树历史未跟踪材料，补交报告和归档说明；真实执行结论随后追加。下方未批准为先前时点。

## 2026-11-19 NAV-067首个受控病例真实失败，已停止并准备隔离恢复

准备版本先提交`a70214e`，随后在tiger11原tmux会话`nav064-r14-repair-20261115:nav067-remaining-plan`（%600）、GPU6启动剩余包。C01在创建环境后、warmup/测量循环前报`actual clone origins are not separated`，按原失败规则停止。新增应用1、受控通过0、额外导航回合0；C02–C06和六次导航重复未启动。初始化物理步未完整记录，不能把未到达测量循环写成总物理步0。

失败应用清理阶段已终止，进程已释放；即使退出码为0，协调器仍根据ERROR和缺失FINISHED保存HELD，未继续或自动重试。输出`/pub/data/caohy/work/camra_navigation/nav067_remaining_validation_v1`原样保留；实际累计GPU进程时间118.444秒，失败槽位408环境子步仍计入预留。完整异常、hash和局限见[失败审阅](../reports/nav067_remaining_validation_v1_execution_review.json)及`.worklog-testing/NAV-067/remaining-v1/`。

根因已用实际安装源码核对：准备误读`/pub/data/caohy/IsaacLab/`，实际解释器导入`/pub/data/caohy/IsaacLab_v0472/`。该版本generator不支持新增原点开关，总从单地形单元分配，导致双环境原点不分离。异常前实际原点数值未保存；CPU复算只能说明接口行为，不能充当旧运行的实际原点证据。此失败属于夹具/环境接入，不判导航策略性能失败。

新增`controlled_terrain_v2.py`、`nav067_controlled_simulation_v2.py`及`run_nav067_remaining_recovery_v2.py`，只为人工受控夹具改用当前版本支持的plane/grid原点接口，并在拒绝前保存原点。原v1五份源码/执行清单/批准、地图/目标/位姿时序、导航重复生产链、原六条pilot和固定v2整套均不改。主agent26项CPU基本检查通过，包含实际安装方法复算和缺批准拒绝；没有新增恢复应用，不是实际物理验收通过。

[恢复提案](../config/nav067_remaining_validation_recovery_proposal_v2.json)为PROPOSED，pin `e8b70cb9d7c81177d576189772b5aecb6054410df71dde867aa5ad1c4c034cac`，APPROVED登记尚不存在。请求只恢复C01一次到原目录`C01/attempt2`，之后按原顺序执行未启动项；累计启动上限从9改10，仍一张GPU6串行、累计4小时含失败118.444秒、60000环境子步，精确含失败预留57830。首次新实际证据/入口/运行/轨迹失败仍保存停下，无自动再试。原批准不覆盖本恢复例外，不能重启旧协调器；先取得明确确认、另存批准/pin并核验后才能用v2入口。详见[通俗恢复说明](nav067_remaining_recovery_v2_20261119.md)。

NAV-067保持false、54/67，原G4/效率FAIL和已接受局限保留；不训练、不查询W、不执行NAV-043/final_*、不改共享框架或冻结资源、不推送。受控和重复性实际证据仍须补齐，不能以26项基本检查替代NAV-067六项验收及全局生产链一致性。以下是保留的先前时点。

## 2026-11-19 NAV-067剩余包已批准、隔离实现和基本检查已封存

用户确认了6短病例及6次同配置导航重复，主agent按legacy执行，无测试子agent。新增`controlled_cases_v1.py`、原生物理夹具入口、被动碰撞体几何读回、保存轨迹审计和串行协调器；原physical_v2/clean_v3/nav067_simulation_v3及冻结v2全部保持hash。C01–C05明确为写入测试位姿后读真实物理状态的事件检查，C06要求base到指定障碍的原生过滤接触力，地面支撑力不作替代。

基本CPU检查、24项既有相关测试、原六条轨迹复核通过，证据在`.worklog-testing/NAV-067/remaining-v1/`。首次GPU0显存不足的预检保留，封存前固定空闲GPU6；新应用/物理步仍0，不是实际门禁通过。执行清单pin `5e7fc139288177c7675c486bb4eec5dafb8971ce9e58c18c7dc1788912ef2475`，绑定具体夹具/时序/容差/源文件/测试证据；APPROVED登记与历史PROPOSED分开保存。

下一入口为`run_nav067_remaining_validation.py`，tiger11/tmux `%600`，绝对Python3.11.15；输出只用`/pub/data/caohy/work/camra_navigation/nav067_remaining_validation_v1`。固定一张物理GPU6、串行C01–C06→L1/L2/L3，最多9应用/4小时/60000环境子步，精确槽位上限57422，不转移余量。任何实际受控证据/入口/运行/轨迹失败均保存停下且不重试；受控全过才重复，重复业务失败保留。NAV-067=false、54/67，NAV-043未执行。以下为历史准备时点。

## 2026-11-19 NAV-067剩余验收提案已准备，未执行新增物理测试

在已提交六回合接线结果的基础上，形成[说明与资源表](nav067_remaining_validation_proposal_20261119.md)及[机器提案](../config/nav067_remaining_validation_proposal_v1.json)，状态PROPOSED。建议仅补6个短受控病例及原6配置各1次重复，单GPU累计上限4小时、最多9次应用启动；包含warmup的上限为300环境秒/60000环境子步。病例覆盖中间碰撞后退出、危险历史、转弯足迹、同刻事件、批内异步和原生接触；人为位姿夹具与自然接触证据分列。

主agent在tiger11/tmux `nav064-r14-repair-20261115:nav067-remaining-plan`（%600），使用绝对Python3.11.15完成协议身份、额度算术、原任务定义/历史记录不变及legacy结构核验，见`.worklog-testing/NAV-067/remaining-proposal-v1/verification.json`。不是实际受控门禁通过：入口尚未实现，新增应用/物理回合/训练/W查询均0，无测试子agent。NAV-067仍false、54/67，NAV-043未执行，原G4/效率FAIL、旧pilot及固定v2全部保留。

下一步需要明确批准这包新增资源及实施范围；之后先实现隔离入口、基本检查、封存具体夹具/时序/容差/hash和独占槽位，再按C01–C06→L1/L2/L3顺序执行。任何受控证据缺失/运行错误保存并停下，无自动重试、不挪额度、不把业务超时/超支改写为成功，不访问final_*或推送。旧恢复登记明确排除了受控/重复，这次提案不能自行视作批准。以下是已结束六回合及更早的历史时点。

## 2026-11-19 NAV-067已批准一次恢复的实际结果

NAV-067一次恢复结束：6回合接线轨迹核验完成，受控/重复验收尚未完成。本次保存6个实际回合结果，6个通过接线轨迹核验。状态`PILOT_TRACE_PASS_NAV067_INCOMPLETE`，不以退出码或基本测试代替实际证据；故障及原L1 attempt1全部保留。批准登记、原manifest和295项保护身份匹配，无训练/W查询/自动重试。

| 地图 | 方法 | 结束原因 | 时间(s) | 安全历史y | 实际机械功(J) | 共同预算(J) | 实际超支 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| l1_empty-s3001 | full | goal_reached | 39.030 | 0.0 | 426.473 | 340.848 | 25.1% |
| l1_empty-s3001 | uncalibrated | timeout | 45.000 | 0.0 | 453.423 | 340.848 | 33.0% |
| l2_single_detour-s3001 | full | timeout | 45.000 | 0.0 | 509.394 | 413.776 | 23.1% |
| l2_single_detour-s3001 | uncalibrated | timeout | 45.000 | 0.0 | 492.399 | 413.776 | 19.0% |
| l3_multi_path-s3001 | full | goal_reached | 44.510 | 0.0 | 482.318 | 340.961 | 41.5% |
| l3_multi_path-s3001 | uncalibrated | timeout | 45.000 | 0.0 | 473.726 | 340.961 | 38.9% |

以上为共同预算接线pilot，非W所选预算的正式物理性能；超时/超支保留，不能写成安全可行成功，也不能据此证明所有策略均无解。

实际审阅见[本轮报告](../reports/nav067_recovery_v3_execution_review.json)和`.worklog-testing/NAV-067/recovery-v3-execution/`。NAV-067仍false、54/67；真实受控反例及同配置重复未获额外额度，NAV-043未执行。以下为此前时点。


## 2026-11-19 NAV-067接口回修v3：补齐刚体读回与直接循环复位初始化

继续只读核对实际IsaacLab0.47.2源码，发现旧循环还有两处确定的接口阻断：RigidObject的RigidBodyView应读`get_transforms()`，旧入口误用Articulation的`get_root_transforms()`；ManagerBasedRLEnv的`reset_buf`在常规`step()`才首次创建，直接子步循环不能提前读取它。新增`clean_v3.py`只补齐本次环境实例的复位记账，并按实际刚体接口读回；`nav067_simulation_v3.py`从封存函数独立派生，只有适配器导入和刚体读回两处变化。新入口为`evaluate_navigation_v4.py`，旧四源码、v2恢复提案及此前入口保持原样。

主agent37项基本测试通过，另实际调用生产入口确认缺少批准登记时在创建目录/应用前拒绝。295项保护身份及旧清单不变；本轮新应用/物理步/导航回合/训练/W查询均0，未派子agent。这些证据只支持接口回修准备，不是实际Isaac物理接入通过，也不还原首次启动未保存的异常栈。

[最新恢复提案v3](../config/nav067_startup_recovery_proposal_v3.json)替代v2作为未来执行依据，仍为PROPOSED；同一L1一次恢复加原L2/L3四槽、最多6个45秒回合，额度和评分不变。批准登记应另存`nav067_startup_recovery_v3.json`及pin；用户尚未确认，不能运行。原失败、旧版本/提案/hash全部保留；受控真实反例与同配置重复额度仍未分配，NAV-067保持false、54/67，NAV-043未执行。详见[准备审阅](../reports/nav067_api_repair_preparation_v3.json)及`.worklog-testing/NAV-067/api-repair-v3/`。

## 2026-11-19 NAV-067启动回修准备：路径缺陷已定位，恢复方案待确认

原启动环境的`find_spec('unitree_rl_lab')`为null，但仓库内对应源码在位；旧生产函数在创建gym环境前会导入该模块，因此存在确定的入口路径缺陷。新增隔离`nav067_runtime.py`固定本仓库源码来源，并在Kit关闭前持久保存异常；`evaluate_navigation_v3.py`复用封存的v2仿真函数，保持旧四份源码及清单不改。29项主agent基本检查通过，覆盖关闭应用退出时异常仍能落盘和未批准恢复拒绝；无测试子agent。

另做一次零机器人环境、零物理步的实际Kit启动/导入/配置检查，PID1297704于08:19:58退出：补齐路径后，Isaac应用、任务模块、clean配置均加载成功，实测配置dt=0.005/decimation=4、资产路径匹配；不能替代真实机器人/碰撞体/命令/计量验证。材质插件错误在该成功检查中仍出现，未改共享环境，也未据此断言对实际物理无影响。首次异常未在清理前记录，不能虚构原始异常栈或把所有启动问题都归因于一个模块。

[恢复提案](../config/nav067_startup_recovery_proposal_v2.json)为PROPOSED：L1在原目录内新增attempt2，恢复原两方法；L2/L3沿用原未启动四槽；最多执行6个45秒导航回合，保留首次失败、不自动再次重试，不训练/搜索/访问final_*。由于原封存清单禁止自动重启，本次只完成可审阅修复和恢复提案，没有生成批准登记或执行恢复。原L1两个槽仍占用，实际回合0、训练0、W查询0；真实受控反例和同配置重复额度不包含在本提案。NAV-067仍false、54/67，NAV-043未执行。

证据：[准备审阅](../reports/nav067_startup_recovery_preparation_v2.json)、`.worklog-testing/NAV-067/startup-recovery-v2/`的preflight、probe原始日志/RESULT及基本测试。下一入口是确认一次L1启动恢复例外后，另存APPROVED登记及pin，再由v3入口执行原清单并逐条核验；未获确认不能把PROPOSED当批准。

## 2026-11-19 NAV-067首次启动未产生有效轨迹，保留attempt1

代码和清单已在本地提交`28f5b7c`、`22a23cc`，未推送。随后在tiger11/tmux `nav064-r14-repair-20261115:nav067-clean-v2`（@587/%587）启动L1的attempt1，PID1232883于08:11:44退出，退出码0；只有STARTED及启动日志，没有runtime、回合结果、200Hz轨迹或FINISHED，不能作为物理接线通过或策略失败。日志确认Isaac材质插件缺失并报`omni.UsdMdl`导入错误；Kit之后仍到达app ready再关闭，因此尚未确认缺少有效轨迹的直接根因。

原始启动记录的tmux窗口字段因未指定pane错误写为`nav065-v11-preparation`，原记录保持不变；以@587/%587的pane记录另行更正。L1两个槽仍占用且不自动释放，L2/L3四个槽未启动；完成的有效物理回合为0，不以此推断所有初始化物理工作均为0。已保留原启动/退出记录、原始日志副本及Kit日志hash，见`.worklog-testing/NAV-067/entry-v1/startup-attempt1/verification.json`。本次没有新增训练、W查询、试跑或修改冻结环境/源码/清单。

NAV-067仍false、54/67。下一步先定位启动/入口故障，必要回修保留原版本，明确恢复和额度后再执行；不自动重跑L1或消耗L2/L3。真实受控反例及同配置重复的独立额度仍未解决，NAV-043及final_*不开启。以下为启动前的历史时点。

## 2026-11-19 NAV-067新版生产入口准备：24项基本检查、295项保护身份核验

已新增隔离入口`navigation/scripts/evaluate_navigation_v2.py`及身份/额度绑定`nav067_sources.py`，使用NAV-066已选两模型、原45秒/obs-v7和固定v2。新的物理链明确使用原归一化动作映射、实际50Hz重新观测和200Hz子步事件/机械功，绕过gym自动复位。主agent24项基本检查通过，实际CPU重载两套actor/双critic及295项保护输入hash匹配；未派测试子agent。代码检查不证明真实Isaac接入通过，NAV-067仍false，54/67。

唯一pilot清单已封存于`/pub/data/caohy/work/camra_navigation/nav067_clean_pilot_v1/manifest.json`，三图seed3001各两方法，共6回合，来自NAV-061既有clean配额。沿用原接线共同预算k=1.0，按冻结45秒独立派生的参考为L1 340.847764J、L2 413.776269J、L3 340.961039J，两方法使用相同J；非价值预算选择、非科学性能分母，不冒充独立新地图。每level启动前独占两个槽，不得换根目录或自动重跑。尚无新增物理、训练或W查询。

下一步：在tiger11/tmux现场核GPU后，以绝对Python3.11.15和`PYTHONPATH=src:scripts:../calibration/src`执行`-B scripts/evaluate_navigation_v2.py --level L1_empty --device cuda:0`，随后按清单L2/L3，先检查实际clean/碰撞体/时钟及原始trace；同配置重复和真实受控反例需要单独明确额度。旧NAV-042入口和报告全部保留，不能用其旧候选/30秒默认值替代新版证据。[准备核验](../reports/nav067_pilot_binding_v1.json)及`.worklog-testing/NAV-067/entry-v1/`记录测试、绑定和旧协议解析失败：原冻结协议在无关总数字段含算式，保留原文及hash，新读取器仅取唯一已登记6回合标量。以下为历史时点。

## 2026-11-19 用户接受预算效率局限，下一入口NAV-067

用户同意此前建议：接受已知预算效率局限，继续研究流程，暂不围绕单例反复重训。[独立处置记录](../reports/nav066_efficiency_limitation_acceptance_v1.json)绑定原注册和结果hash；NAV-066按`ACCEPTED_WITH_LIMITATION`流程结案，原完整方法效率8/9和科学FAIL、旧G4 FAIL、原代码/模型/阈值/报告均不改。计划原位升1.4.1，仅NAV-066 false→true，其余66项、依赖和全部历史记录保持；当前54/67，下一项NAV-067尚未开始。

这次只核对已有90条/18决策结果与保护身份、登记接受并梳理任务，没有新增训练、优化、价值查询、名义或物理回合。接受只解除已测配对一例预算过保守的阶段阻塞，不是物理/最终性能认证，资源配额仍须逐项现场核对，final_*继续封存。后续见[简明任务路线](nav066_accepted_limitation_next_steps_20261119.md)：NAV-067物理接线→043闭环适用性→044正式冻结→045 clean→046/047扰动→048–050复算评审→051–054交付。

重要版本边界：新版模型45秒，NAV-043旧文本30秒及旧pilot默认候选不能直接复用，后续开工需绑定本轮模型、双critic、预算规则与任务契约；L2最慢44.7秒，物理时限余量须实测。G5/G6必须同时披露原G4 FAIL、本轮独立效率FAIL和明确用户接受处置，不能写成旧门禁重验PASS，也不豁免物理适用性/安全/能耗/数据锁。以下停止/false为历史时点。

## 2026-11-18 NAV-066 独立检验结束：任务全成功，预算效率仍有一例不通过

使用已固定最好现有配对完成90条独立名义网格轨迹/33531步，全部18个预算决策在轨迹前封存。两方法各9/9按所选预算45秒内安全且预算内到达，错误放行/拒绝均0，三level各3/3。无校准PASS；完整方法9图选1.30，其中8图真实最低1.15，一图L1 `l1_empty-s917001`实际1.00可行却预测不可行，加一档保护后预留两档，效率8/9、方法FAIL。该例不是危险或超支，所选预算比一档效率上限多约60.81 J；不能把预留多的预算写成实际多耗同样的能量。

NAV-065新版只研究准备true，NAV-066保持false，总53/67，NAV-067不启动；原G4 FAIL及拟合历史保留。13项基本测试及主agent全轨迹/J/安全/时间/来源复核通过，不改变科学门禁结果；训练/更新/物理/final访问0，无测试子agent。90回合和90查询额度已用完，不能自动重跑或按本次数据改阈值。下一步先评审此保守价值估计与留量叠加的局限，任何新方案按新版本/隔离数据/额度处理，不默认再训练。

详见[通俗审阅与完整证据](nav066_budget_decision_review_v2_20261118.md)、[机器报告](../reports/nav066_budget_decision_validation_v2.json)。原始证据`/pub/data/caohy/work/camra_navigation/nav066_budget_decision_v2_best/`包含90条原始轨迹/全部封存查询/hash/FINISHED，执行及只读复核日志`.worklog-testing/NAV-065/best-decision-v2/`。事前代码协议提交`0696350`，无推送。下文“待执行/未激活”都是保留的历史时点。

## 2026-11-18 新协议已激活：最优现有模型独立决策验证

用户确认采用现有最好模型检验；已按既有development固定无校准seed2 guarded-v4、完整方法seed0 original，见[身份与完整历史排序](../reports/nav065_best_existing_model_binding_v2.json)。原G4 FAIL、训练错判及旧标准保留，本轮不再以训练起点清零为默认方向。主计划1.4.0原位分离NAV-065准备与NAV-066实际决策，67项顺序/依赖不变。NAV-065仅研究准备通过，53/67；[准备证据](../reports/nav065_budget_decision_readiness_v2.json)包括六组CPU张量核验、13基本测试和9图准入，新查询/轨迹均0。

[批准登记](../config/nav066_decision_prereg_v2.json)冻结五档W查询加一档预算保护，异常/非单调/无留量拒绝。全部18决策先封存，再执行90条独立名义网格轨迹；选中行复用，每方法9个实际决策，不能声称90独立决策。按方法检查零放行失败、80%可行图召回、至少6/9放行且各level至少2/3，以及不超过真实网格最低可行档一档。当前独立检验待执行，不预告通过。

旧草案30秒系笔误，本次保持训练模型原冻结45秒任务（含观测及E_ref）；90回合上限不变，最多40500名义步。新917001–003任务实例/几何与历史分片互斥，布局共享；final_*封存。输出目录`/pub/data/caohy/work/camra_navigation/nav066_budget_decision_v2_best`，执行入口`scripts/run_nav066_decision_validation.py`，tmux `nav064-r14-repair-20261115`，Python `/pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python`。主agent按legacy核验、不派专用测试子agent；更新/物理回合0，失败不自动重跑或调模型。下方未激活/30秒/旧来源为历史时点，由本次已批准登记限定覆盖。

## 2026-11-18 NAV-065 新版预算决策验收草案：停止默认重训，尚未激活

用户批准先复审验收设计。已写入[新版说明](nav065_budget_decision_protocol_v2_20261118.md)和[机器协议](../config/nav065_budget_decision_protocol_v2.json)：全量训练误差、全部边界错判保留诊断；拟用固定双critic五档查询、首个预测可行档再加一档预算，无留量/零点/非单调/非有限值返回null。经验保护不称理论保证，原目标、beta=0、v2/alpha/delta不改。

建议每方法实际放行失败0次、超支0 J、时限30秒；真实可行图召回至少80%，至少放行6/9图且各level至少2/3；所选预算最多高于同actor真实网格最低可行值一档。这些是研究用途的待确认建议，不以旧13次误判反推线，不改写旧G4。拟先用v7完整六候选统一阶段，按既有最小策略种子规则只测两方法seed0，具体checkpoint/hash已绑定；v7原train失败和缺少匹配新版development通过的事实保留。

旧开发及旧选择集均已暴露，建议派生917001–917003×三level独立任务清单，final_*封存。只检查了种子元数据互斥，未生成新图或证明任务实例独立；执行前须完整历史暴露/几何/参考成本审计。拟最多90名义回合、27000步、最多18预算决策（每方法9），更新/物理回合0；先封存全部选择再执行，不把90网格行写成90独立决策。通过也只是单种子名义筛查，不是总体风险或物理证书。

本轮主agent按legacy作基础检查，无专用测试子agent；证据见[准备复核](../reports/nav065_budget_decision_protocol_v2_preparation_review.json)和`.worklog-testing/NAV-065/decision-protocol-v2/`。协议未激活，生产入口尚未实现新版保护；原计划1.3.1、52/67、NAV-065=false、原G4 FAIL、NAV-066未启动保持。下一入口是整体确认草案数值、来源、独立数据和90回合额度，再原位修订NAV-065准备验收与NAV-066实际决策验收，保留任务依赖。当前不训练、不执行独立集、不挪用已结束额度。

## 2026-11-18 NAV-065 v11结束：稳住全量，关键误判仍未减少

用户确认v11具体提案后在tiger11/tmux完成一次full_s0副本验证，两原输出层各求解一次/共2次，梯度更新/新回合/开发评分0。复用162条train/59877步，原主预算实际可行及识别18/18保留；起点误放行13→13、误拒绝1→1，新增到达相对MAE43.85%→41.31%、Pearson0.5722→0.6129，边界联合错误97→94。原全时刻W相对MAE6.46%→20.96%、起点到达符号错误28→34，仍有退化，完整train FAIL。legacy主agent原生产门禁/CPU重载/仅四head变化/六原模型及33项v2复核通过，38基本测试与未变源码匹配，无测试子agent。原G4 FAIL、52/67、NAV-065=false及NAV-066未启动；2次额度已结束，不追加或自动扩展。

详见[本轮结果](nav065_ridge_probe_v11_review_20261118.md)、[机器审阅](../reports/nav065_ridge_probe_v11_review.json)及[已批准登记](../config/nav065_ridge_probe_prereg_v11.json)，pin为`ea32c6e299ea68fbd53c7e0e901669262e71dc44f9f49ec5386821e3951f45aa`。原始产物`/pub/data/caohy/work/camra_navigation/nav065_ridge_probe_v11/`；tmux `nav064-r14-repair-20261115:nav065-v11-execution`，日志`.worklog-testing/NAV-065/ridge-v11-execution/`。执行完整性通过不等于科学通过；下一步先诊断已有train特征/符号/拟合折中，不自动开始新拟合。下方准备和未批准措辞保留为历史时点。

## 2026-11-18 NAV-065 v11准备：全量共同拟合与输出层稳定性

用户指定create-plan-worklog-legacy并要求按前述分析推进，主agent完成38项基本检查，无测试子agent；固定v8末次full_s0来源，162条已存train/59877步通过只读生产接线预检。原33223步、已有新增26654步、162起点和原866个边界行按四组/到达符号分层加权，共60905个行出现；没有新增采样。只读包装器禁用候选求解、Adam/SGD、rollout和development评分。

[隔离v11入口](../scripts/probe_nav065_ridge.py)对全量原数值目标做单次增广最小二乘，加入固定0.01的相对源输出层改变量惩罚，不改共同停止、W、beta=0、actor/body、架构或原v5 train/G4。按正则目标不高于源的条件，成本/到达改变量界约0.05296/5.44044，明显小于v10精确插值的实际42.47/552.04；这个界不保证逐组误差、符号或泛化，不能作放行标准。

[具体提案](../config/nav065_ridge_probe_proposal_v11.json)未批准：一个副本，两个原输出层各一次新求解、总2次；梯度更新/新回合/开发评分0，不扫参数、不扩展六候选、不生成部署包或新G4。真实拟合须明确确认并新建批准登记/hash；v10已结束，不挪用其额度。正式结果仍直接调用原v5全量train_gate，并报告原25点和全部起点/边界/组误差的退化。

现场tiger11/tmux `nav064-r14-repair-20261115:nav065-v11-preparation`、Python3.11.15绝对解释器及33项v2资源核验匹配；继续用户已确认的同机执行例外。开工辅助脚本的混合换行解析失败保留在context.log，独立open.py仅恢复追加，全部任务/历史保持。准备和主agent复核见[v11说明](nav065_ridge_probe_v11_20261118.md)、[只读预检](../reports/nav065_ridge_preflight_v11.json)及[准备审阅](../reports/nav065_ridge_probe_v11_preparation_review.json)。原始日志在`.worklog-testing/NAV-065/ridge-v11-preparation/`。

本轮真实求解/梯度更新/新回合/开发评分均0；原13次误放行、G4 FAIL、52/67、NAV-065=false及NAV-066未启动保持。下一入口是评审具体2次全量正则拟合额度；准备通过不表示科学通过，下方v10等结果为历史时点。

## 2026-11-18 NAV-065 v10结束：局部可拟合，全量退化

用户确认2次具体求解后于tiger11/tmux 20:57:40–20:57:57完成一次full_s0副本诊断，两个原输出层各一次float64线性求解并转回float32，固定actor/body。实际求解2、梯度更新0、新回合0、开发评分0；不扫参数、不重试。25起点成本/到达/联合符号错误11→0/9→0/13→0，局部门禁PASS，证明这些已见点在实际网络可拟合。

全量仍FAIL：162起点误放行13→13、误拒绝1→58，原90回合主预算18个实际可行起点正确识别18→2；原90回合到达相对MAE15.65%→11946.00%，旧新增72回合43.85%→10794.54%，所选866沿途边界错判97→439。原六模型/actor/固定v2不变，原18/18实际可行未改写，退化仅属于未部署的诊断副本。不能把25点通过写成NAV-065通过，也不能由此断言全任务表达充分或必须换架构。

legacy主agent重建162条train/59877步，复核原局部和原v5全train生产门禁、float32预测、CPU重载、仅四head张量变化、原六模型和33项v2身份；审计额外求解0，准备阶段33项基本检查与绑定源码一致，无测试子agent。详见[结果](nav065_head_probe_v10_review_20261118.md)、[机器审阅](../reports/nav065_head_probe_v10_review.json)和[独立批准登记](../config/nav065_head_probe_prereg_v10.json)，pin为`ab7962f3fe8d1bccfd1498293a277c3065c773fe651366eedd73986879378b21`。原始产物`/pub/data/caohy/work/camra_navigation/nav065_head_probe_v10/`，启动/复核日志`.worklog-testing/NAV-065/head-v10-execution/`，checkpoint仅供诊断。

本次2次额度已结束，不追加或重启；原G4 FAIL、52/67、NAV-065=false及NAV-066未启动。下一步应依据现有证据分析原轨迹/边界/沿途共同拟合的稳定性，未批准新的拟合、开发评分或架构变化。下方准备/待批准措辞为历史时点。

## 2026-11-18 NAV-065 v10准备：先分清表达、数值与优化

用户指定legacy并要求基本验证，主agent完成33项检查，无测试子agent。固定v8末次full_s0，复核该actor的162条已保存train/59877步、原完整train_gate及v9的25起点；核验v9的400次更新、401次评估和原始散列。25点末层特征在rcond=1e-12下均行秩25、目标投影残差近零，但v8成本/到达精确插值的最小改变量范数约42.47/552.04，当前末层范数仅0.330/1.324。此证据支持先查优化与数值稳定性，不能推出通过35%误差线必需这么大改动、float32必定成功、全任务观测充分或必须换架构。

[隔离入口](../scripts/probe_nav065_head.py)与[未批准提案](../config/nav065_head_probe_proposal_v10.json)备妥：一个full_s0副本，固定actor及两个critic body，仅两个原输出层各一次float64最小改变量线性求解、总2次，转回float32按原网络判断；不扫参数、不重试、不换目标或架构。直接调用原v9局部检查和原v5完整90+72 train_gate，报告全量退化，CPU重载并核验非head张量及原模型不变。即便局部通过，也不授权开发评分、G4、部署或NAV-066。

本轮真实候选求解/优化更新/新回合/开发评分均0，33项v2身份匹配；只读预检和准备复核通过。执行在tiger11/tmux `nav064-r14-repair-20261115:nav065-v10-preparation`，使用Python3.11.15绝对解释器，继续用户已批准的同机执行例外。详见[v10说明](nav065_head_probe_v10_20261118.md)、[只读诊断](../reports/nav065_head_preflight_v10.json)及[准备复核](../reports/nav065_head_probe_v10_preparation_review.json)，原始日志在`.worklog-testing/NAV-065/head-v10-preparation/`。

下一入口是评审这次具体2次求解额度；真实候选求解属于新拟合，不能以优化0步绕过批准，也不挪用已结束v9额度。原G4 FAIL、52/67、NAV-065=false及NAV-066未启动保持。下方结果和待执行措辞均为历史时点。

## 2026-11-18 NAV-065 v9结束：局部未过、全量估值退化

用户确认v9后在tiger11/tmux完成一次full_s0副本25起点局部拟合，实际400更新、新回合/开发评分0。局部误放行13→4、成本符号错误11→4，但到达符号错误9→9、相对MAE69.86%→44.82%仍未到35%线；局部FAIL。诊断副本原90回合全时刻到达相对MAE15.65%→293.37%、旧新增72回合43.85%→140.95%，完整train仍FAIL。六原模型/actor及33项v2身份不变，原主预算18/18实际可行与起点识别保留；未替换部署模型。legacy主agent逐步/原门禁/CPU重载复核通过，无测试子agent；执行完整性不等于科学通过。原G4 FAIL、52/67、NAV-065=false和NAV-066未启动保持；本次尝试结束，不自动重启、扩展或加跑。

详见[v9结果](nav065_local_probe_v9_review_20261118.md)。以下为历史时点。

## 2026-11-18 NAV-065 v9准备：局部可拟合性诊断

用户指定legacy按前述分析推进；主agent完成16项基本检查和v8末次162条train/59877步只读核验，无测试子agent。原13次误放行分为9次到达分量应拦截、4次仅成本应拦截；固定13错例及相邻真实可行点去重共25起点。到达末层特征在报告容差下行秩25，不等于全任务观测充分或保证拟合成功。隔离v9入口和未批准400步单候选局部拟合提案已备妥；正式更新/新回合/开发评分0，原目标/架构/actor/v2及原v5 train/G4不变。准备通过不等于部署通过；原G4 FAIL、52/67、NAV-065=false和NAV-066未启动保持。新额度待明确确认，不重用v8余量。

详见[v9方案](nav065_local_probe_v9_20261118.md)。以下为历史时点。

## 2026-11-18 NAV-065 v8结束：单候选验证结果

用户确认v8后在tiger11/tmux完成一次固定full_s0单候选验证，legacy主agent验收，无测试子agent。复用该候选162条train/59877步，新增回合/开发评分0；实际数值300+边界0、共300更新。临界数据到达相对MAE 80.35%→43.85%（原线35%），Pearson 0.3312→0.5722（原线0.70）；起点误放行3→13、误拒绝10→1。停止原因numeric_gate_not_met，pilot完整train FAIL；六actor及其余五模型不变、33项v2身份匹配。无新G4/部署包，原G4 FAIL、52/67、NAV-065=false及NAV-066未启动保持；本次单次尝试已结束，不自动扩展或挪用阶段余量。

详见[v8结果](nav065_pilot_refit_v8_review_20261118.md)。以下准备/未批准记录保留为历史时点。

## 2026-11-18 NAV-065 v8准备完成：单候选数值优先，正式优化待新额度

按用户指定legacy由主agent完成57项基本检查，无测试子agent；972条train/372471步只读预检、原v5门禁与v7末次结果逐项匹配、33项v2身份核验通过。各固定actor的全部float32输入未发现完全重复，因此未发现同输入冲突标签，但不能证明观测充分。v7到达critic边界900步中874步与回归方向相反，陡变目标和优化冲突证据保留，19次错误放行/64次误拒绝未解决。

隔离v8入口先要求原数值检查通过，再用保留回归方向、只投影并限幅纠错的无动量SGD；原actor/目标/W/beta/v5 train/G4及v2保持。[方案](nav065_pilot_refit_v8_20261118.md)与[未批准提案](../config/nav065_pilot_refit_proposal_v8.json)固定full_s0、最多300数值+100边界/共400新更新，新增回合与开发评分0；即使pilot通过也不自动扩展或部署。本轮正式更新0，无正式run/新批准登记；新过程及数字额度待确认，不使用旧余量。NAV-065=false、52/67、原G4 FAIL、NAV-066未启动。下方为历史时点。

## 2026-11-18 NAV-065 v7结束：两阶段回修未通过完整train

用户确认v7方案后在tiger11/tmux完成一次独立两阶段critic回修，legacy主agent验收，无测试子agent。复用972条train/372471步，新增回合0；六候选数值+边界步数分别150+150/150+150/150+150/150+150/150+150/150+150，共1800次更新，六actor不变、33项v2身份匹配。起点误放行25→19，所选边界错判932→573；旧五档4/6、完整train 0/6通过，状态HELD_TRAIN_GATE。按批准协议开发重评0，无新G4/部署包；原G4 FAIL、52/67、NAV-065=false、NAV-066未启动保持。本次额度已结束，不能自动重启或转移余量。

详见[v7结果](nav065_staged_refit_v7_review_20261118.md)和[主agent复核](../reports/nav065_staged_refit_v7_review.json)。已保存逐步分类变化和阶段模型；先诊断现有证据，不以中间改善选模型替换末次结果。下方未批准/待启动均为历史时点。

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

# 2026-11-19 NAV-067阶段保存：子步基础实现，尚未物理验收

用户要求继续后续任务并核对必要提交。本轮按legacy由主agent验证，未安排测试子agent。在`tiger11`的tmux `nav064-r14-repair-20261115:nav067-clean-v2`新增隔离的`envs/physical_v2.py`及`envs/clean_v2.py`，保留原NAV-042源码与所有已绑定模型/配置。新版基础接口显式使用obs-v7/45秒，逐200Hz实际状态维护不可逆危险历史及物理事件，保存终止步并停止后续计费，部分周期按实际时长计模型成本。Isaac适配器绕开gym.step自动复位，仍通过固定v2 action_manager逐子步施加PD；这部分尚未在真实Isaac环境运行，不能宣称已接线通过。

`tests/test_physical_v2.py`的16项基本测试通过，覆盖中间穿障退出、危险历史、旋转扫掠、同时碰撞到达、异步结束、部分周期及陈旧/缺失命令证据拒绝；这些是合成反例，不替代真实仿真。291项原注册绑定源码/配置/资源/所选模型hash复核一致。报告为[基础准备记录](../reports/nav067_substep_preparation_v1.json)，原始日志在`.worklog-testing/NAV-067/clean-v2/`。

计划仍1.4.1、54/67，NAV-067保持false。新增训练/物理回合/W查询均0；下一入口是补齐新版候选绑定和完整可执行生产入口、运行时clean及碰撞体读回，再在有效登记额度内做6回合pilot。真实受控反例和同配置重复性须核对独立额度，不能静默加跑。原效率FAIL、旧G4 FAIL和接受局限记录保留，不能进入NAV-043/G5或访问final_*。

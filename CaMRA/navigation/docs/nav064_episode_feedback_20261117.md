# NAV-064 完整回合反馈候选（2026-11-17）

后续结果：该候选已获批并完成两方法×三种子各300更新，原始审计通过，开发FAIL_DEVELOPMENT；full安全可行SR中位50%、危险步600，L2训练5400回合安全到达0次。最新结论与继续回修见[绕障探索记录](nav064_avoidance_prior_20261117.md)及`reports/nav064_avoidance_prior_release_v1.json`；下文“尚未启动/待授权”为当时记录，不能重复开旧run。

当前计划1.3.1仍为51/67项历史完成，NAV-064=false。净空版六组训练已结束，开发未通过；本次完成既有数据诊断和独立候选的生产数值验证。没有启动新的正式训练，也未进入NAV-065。

## 已确认的现象及推演

净空版冻结预登记pin为`776b86bfbeb874c5024645653859afb5d121d7b0520b6367a8b2d7dedceb3bb5`。六组各600更新，原始审计通过，最后checkpoint的108回合development结论为FAIL_DEVELOPMENT。主预算k=1.15下，uncalibrated/full的安全可行SR中位数均为1/3，能耗/预算分别0.819125/0.896143，但full危险步中位数780，未满足原关闭档。到达率不能替代安全可行率。

逐条核对full三个种子的全部600更新：L2训练图共1769次到达，安全到达和安全预算内到达均为0（各种子到达712/444/613）。这证明旧采样未提供安全绕障到达的正例，不能单凭到达次数认定策略已学会绕障。

在99/349/599更新选出的36774个实际样本中，首次越界70次，69次被当步较大的距离项遮住；安全历史已饱和的5168个非吸收失败样本均选择当下停止，联合目标为1。这是max与不可逆安全历史的实际行为，不能靠清空历史或删除危险样本解决；当步被遮蔽也不等于所有未来反馈都为零。

用同一初态、同一k=1.15预算，在六张实际L2训练地图执行三种受控动作序列：安全短绕路、危险捷径、缓慢移动直至超时。所有环境均使用原完整几何、功率、终止和45秒时限；危险路径生成器的半δ只用于生成干预动作，实际判定始终使用完整δ。六条安全路径均在400–415步内安全预算内到达，支出约535–561J，剩余约17–22J；捷径到达但带危险历史，缓慢轨迹超时。没有改地图或验收条件来制造可行路线。

这些干预轨迹不是策略采样结果，也不参与正式训练。冻结旧critic后，对比实际目标与一次实际PPO更新：

|估计器/采集|安全路径目标优于两种失败路径|安全起步动作相对密度同时改善|
|---|---:|---:|
|λ=.95，227步切片|8/18|7/18|
|λ=.95，完整轨迹|8/18|7/18|
|λ=.995，完整轨迹|18/18|15/18|
|λ=.999，227步切片|14/18|14/18|
|λ=.999，完整轨迹|18/18|18/18|

18组为6图×3个冻结critic。λ=.95时，完整轨迹终点对首步联合目标的受控敏感度约1e-12，227步切片的直接敏感度为0；λ=.999保留完整轨迹时约6.26e-5至8.27e-5。切片仍可通过学到的bootstrap间接传递信息，不能写成“切片永远不能学”。安全终点联合残差受`-z_T`限制，不能将`h_T=-3`当作联合目标的实际终点数值。

由此支持一个可检验候选：让长绕路结果在同一策略版本下完整进入估计器，并延长GAE的反馈范围。它同时改变采集方式和λ，不是因果消融；训练中如何探索到安全路径仍是未解决的风险。

## 独立候选与实际调用链

配置为`config/value_objective_contract_episode_feedback_v1.json`（1.5.0-episode-feedback）、`navigation_contract_episode_feedback_v1.json`和`training_loop_episode_feedback_proposal_v1.json`。旧契约、旧pin、旧报告和旧训练结果不覆盖。

生产入口`train_formal.py`显式选择`complete_episodes_v1`时，每张train图每次更新采集一个完整的当前策略回合。终止的环境等待下一更新，忽略inactive行；全部18个回合终止后才更新网络，然后统一复位并抽取下一回合预算。提前碰撞、超支、停滞和超时都保留，不补新回合填满样本上限，不携带旧策略前缀。三套估计器共同使用λ=.999；共同停止、minPPO、不可逆安全历史、联合/成本γ=1及reach调度保留。

协调器、原始审计和开发评估消费同一注册版本。候选为每组300更新、每次最多18×450=8100样本；前50更新计入300，先审计50，结束审计300。样本数按实际回合长度记账，8100是上限。旧227步/600更新入口按原规则保留。

当前源码门禁`reports/gate_episode_feedback_v1_attempt2.json`为171/171 PASS，包含重跑的153项上游反馈/时间/净空/运行检查。实际生产入口的两次机械更新在全部18张train图上采样当前策略，保留36回合，共4211+4228=8439样本；180个实际minibatch更新中输入不变，独立回放旧策略log-prob误差为0，两critic采样值也一致。原始审计重算全部36个chunk及三套目标，预算随机流/身份/净空/样本数一致；伪造计数、缺回合和旧更新身份被拒绝。六类终止另经受控终端夹具核验。

机械夹具只在内存注入两次更新的测试注册，目录明确带`onpolicy_mechanical`，没有保存approved预登记，没有用其指标选模型或判性能。正式提案保持`training_approved=false`且无pin，真实启动会拒绝未批准提案。32套回归及冻结资源/历史保全的最终状态以`reports/nav064_episode_feedback_release_v1.json`为准。

## 资源与下一入口

上轮六组额度已用尽。本候选申请两方法×三种子共6个fresh-init run，各300次完整回合更新，最多14580000样本；最后checkpoint做原108回合development，原阈值不变。没有额外性能smoke、自动重试或自动下游执行。批准后才另建不可变预登记/pin，核验资源与输出冲突，提交启动版本并开训。

新增正式次数须按主计划uncertainty_policy确认；CPU门禁通过不授予额外额度。正式训练关闭档、机制/稳定性和数据审计全部通过，才可进入NAV-065。fixed v2整套配置、冻结校准、地图/预算/成本/评分、只读参考和稿件边界保持。

## 复算与证据

执行固定tiger11，沿用已批准同机例外；tmux为`nav064-r14-repair-20261115`，解释器为`/pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python`。资源预检及116份原始源码/配置/计划/报告快照在`/pub/data/caohy/work/camra_navigation/nav064_detour_diagnosis_20261117_v1/`。

受控轨迹在该根的`trajectories_attempt4/`，原始数据/hash索引为`reports/nav064_detour_feedback_v1_attempt4.sources.json`；诊断及λ/窗口比较为同目录`nav064_detour_feedback_v1_attempt4.json`和`nav064_detour_horizon_v1.json`。早期诊断attempt1–3和数值门禁attempt1的失败日志保留，不用最终成功报告改写旧失败。门禁attempt1暴露旧协调器接口夹具未提供window字段，已恢复旧命名空间的路径接口兼容；attempt2重跑上游及生产链通过。

本次机械输出为`/pub/data/caohy/work/camra_navigation/nav064_episode_gate_20261117_a2_onpolicy_mechanical/`，审计在相邻`nav064_episode_gate_20261117_a2/`。日志为`/tmp/nav064_episode_gate_20261117_a2.log`，外部保全索引由release报告定位。重跑必须使用新的`--out`与`--work`，不能覆盖旧目录。

```bash
# 在上述tmux和仓库根目录中运行；复算也必须选择全新输出路径。
/pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python CaMRA/navigation/tests/test_episode_feedback_gate.py --out <new_report.json> --work <new_work_directory>
/pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python CaMRA/navigation/scripts/compare_detour_horizon.py --diagnosis CaMRA/navigation/reports/nav064_detour_feedback_v1_attempt4.json --out <new_comparison.json>
```

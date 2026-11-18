import json
from pathlib import Path

ROOT=Path('/home/caohy/repositories/Paper')
NAV=ROOT/'CaMRA/navigation'
HERE=Path(__file__).resolve().parent
OUT=Path('/pub/data/caohy/work/camra_navigation/nav066_budget_decision_v2_best')
load=lambda p:json.loads(Path(p).read_bytes())
report=load(NAV/'reports/nav066_budget_decision_validation_v2.json')
decisions=load(OUT/'decisions.json')['decisions']
row=next(r for r in report['grid_rows'] if r['method']=='full' and r['map_id']=='l1_empty-s917001' and r['k']==1.0)
d=next(r for r in decisions if r['method']=='full' and r['map_id']=='l1_empty-s917001')
q=d['decision']['trace'][2]
detail=next(r for r in report['methods']['full']['details'] if not r['efficiency_pass'])
ref=detail['budget_J']/1.3
table=[]
for m,name in [('uncalibrated','无校准基线（seed2 / guarded-v4）'),('full','完整方法（seed0 / original）')]:
 s=report['methods'][m]
 table.append(f"| {name} | {s['correctly_accepted']}/9 | {s['accepted_failures']} | {s['rejected']}/9 | {s['feasible_map_recall']:.0%} | {9-s['efficiency_failures']}/9 | {'PASS' if s['pass'] else 'FAIL'} |")
text=f'''# NAV-066 最优现有模型预算决策独立检验（2026-11-18）

这次检验已结束。两方法实际选择的预算都让9/9个任务在45秒内安全且预算内到达，错误放行0、拒绝0；完整方法有1图预算选得过于保守，未通过预先约定的效率条件，因此NAV-066保持false。此次没有训练，也没有改变模型或门槛。

| 方法与固定来源 | 所选预算安全可行 | 错误放行 | 拒绝 | 可行图召回 | 一档效率达标 | 方法门禁 |
| --- | --- | --- | --- | --- | --- | --- |
{chr(10).join(table)}

每方法L1/L2/L3均3/3成功放行。90条是9图×2方法×5档的网格执行，实际预算决策只有18个；选中预算行复用，没有额外重跑。内部消耗按模型账本J计，完整方法包含原冻结成本裕度；这不是物理机械功实测。全部33531名义步，低于40500上限；优化更新/物理回合/最终集访问均0。

## 事前冻结及模型选择

用户确认新版协议并要求“用目前训练出来的最好的模型去检验”。只使用已经暴露的original、refit-v1及guarded-v4匹配development报告，先按主预算正确识别、错误放行、错误拒绝、成本和到达相对MAE总和排序，固定六个run各自最佳版本，再按方法固定最佳种子。完整18来源排序与所有hash见[来源绑定](../reports/nav065_best_existing_model_binding_v2.json)。v5–v11没有匹配的新development正式证据，单候选诊断不代替配对来源。

所测无校准seed2 guarded-v4和完整方法seed0 original的actor/双critic张量未修改。相应旧开发都无起点误判，仅是选择依据，不是独立结果。旧G4 FAIL和历次拟合失败完整保留；新NAV-065只是PROVISIONAL_RESEARCH_READY，不能称价值可靠或科学通过。

[批准登记](../config/nav066_decision_prereg_v2.json) pin `6e783599f76e68cc483bfc9f4d3e7649e4b312f48335a5c7c578f974ab9d126c`。旧草案写30秒是笔误，原模型冻结训练任务实际45秒，观测与E_ref按原任务一致执行；登记在任何新决策或轨迹前保留更正。回合仍90，不增加训练或重复试验，派生步上限40500。

新种子917001–917003与L1/L2/L3形成9个任务实例，几何、冻结足迹/δ、45秒准入、原分片种子/实例与历史JSON元数据暴露审计通过。level布局共享，因此只声称未见任务实例，不声称未见布局或独立同分布。final_*只用原清单元数据作互斥核对，结果封存。

## 实际预算规则和独立结果

实际入口为`scripts/run_nav066_decision_validation.py`，调用`budget/guarded.py`查询同一CPU重载双critic，计算`W=max(V_h,V_c-z0_J/550)`。五档0.70/0.85/1.00/1.15/1.30全部查完，异常、精确零、非单调、无解或无留量均拒绝；首个严格预测可行档再加一档，标签不参与选择。

全部18个决策及90次初始W查询先封存，再执行任何轨迹。后续每图先跑选中档，再跑其余四档作事后召回与效率诊断；拒绝也不能靠真实标签重新选预算。独立oracle重建每条轨迹的时间、不可逆安全历史、原共同停止、预算J账本和可行性，与实际事件结果一致。

无校准9图均选择k=1.15，真实网格最低可行均k=1.00，恰好多一档保护。完整方法9图均选择k=1.30：其中8图真实最低为1.15，保护符合一档要求；另外1图实际1.00就可行，所以多了两档。未以全部给最大预算自动判通过。

原始W混淆：无校准45行中27真可行、18真不可行，误放行/误拒绝0/0；完整方法18真可行、26真不可行、0误放行、1误拒绝。这里的“误拒绝”是该图k=1.00原始预测，不是最终拒绝整个任务；最终9图均接受。预算保护后各9个实际决策都成功，不能把原始网格错判与最终错误放行混为一谈。

## 唯一未达标项：L1起点预算估计过于保守

图`l1_empty-s917001`（L1空图）完整方法在k=1.00实际安全可行：预算{row['budget_J']:.3f} J，内部消耗{row['cost_model_J']:.3f} J，到达{row['t_s']:.1f}秒。然而该档初始critic输出`V_c={q['vc_norm']:.9f}`、`V_h={q['vh']:.9f}`，联合`W={q['W']:.9f}>0`，被预测为不可行。

所以固定规则从预测首可行1.15再加一档，选择1.30（{detail['budget_J']:.3f} J）；这次实际消耗{detail['cost_model_J']:.3f} J，{detail['t_s']:.1f}秒到达，未碰危险、未超支。事前效率门槛规定最多比真实网格最低多一档，即本图最多1.15（{1.15*ref:.3f} J）。所选预算比该上限多{.15*ref:.3f} J，效率失败。

大白话：这次没有“钱给少了导致任务失败”，而是“已经够用的一档被估计得太保守，再叠加一档留量，钱预留多了”。这不是实际多花了同样的J数，也不是全局最短路径证明。只凭这1图不能推断所有场景的唯一根因，但当前链路中的这项失败可以由具体critic分量与固定规则完整解释。

完整方法L2选中档到达43.6–44.7秒，均在45秒内，最小时间余量约0.3秒；名义可行不保证物理执行有足够时限余量，后续NAV-043仍须检查，不能用本轮跳过。

## 验证、保存和后续边界

主agent按legacy完成13项基本测试，无专用测试子agent。六组CPU重载/张量校验、33项v2身份及原契约保护通过；正式检验后重新核对90个唯一网格键、180条start/result日志、33531步、决策先封存、初始观测相同、J/安全/时间/oracle、每图先选中档及全部评价分母。复核不产生新W查询、训练或轨迹。

首个审计程序在最终报告写完前启动，因文件尚不存在退出；日志`audit.log`保留。完成标记出现后仅重启只读审计，`audit-finished.log`与`final-audit.json`通过；未重跑任何业务轨迹，也不把此文件时序问题算作策略失败。首次提交检查发现README尾空行，修正后提交；原失败检查日志同样保留。

协议/代码事前提交`0696350`。原始目录`/pub/data/caohy/work/camra_navigation/nav066_budget_decision_v2_best/`保存owner、manifest、readiness、decisions及hash、episodes、decision_validation、FINISHED。机器结果见[完整报告](../reports/nav066_budget_decision_validation_v2.json)；主agent复核及全部执行命令/日志在`.worklog-testing/NAV-065/best-decision-v2/`。执行固定tiger11/tmux `nav064-r14-repair-20261115`、Python `/pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python`；未推送。

本次准备已完成（NAV-065=true），独立验证按原门槛未通过（NAV-066=false），计划53/67，NAV-067不执行。后续先评审这1例保守估计及留量叠加的局限，保留本轮数据为已暴露独立检验；不默认重训、不按本次结果反调门槛。若要实质改变规则或再检验，须新版本/独立数据/额度。通过的安全与召回证据完整保留，不能因为效率失败就写成策略不会到达，也不能因为9/9成功就称全部验收完成。
'''
(NAV/'docs/nav066_budget_decision_review_v2_20261118.md').write_text(text)
print(json.dumps({'failed_case_k1':row,'critic_at_k1':q,'selected':detail},ensure_ascii=False,indent=2))

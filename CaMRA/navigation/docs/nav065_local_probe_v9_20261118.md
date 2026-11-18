# NAV-065 v9：先验证已见临界预算能否局部拟合

本项沿用计划1.3.1的NAV-065，主agent按create-plan-worklog-legacy完成基本验证，不安排测试子agent。NAV-064主预算名义导航结果保留；当前52/67、NAV-065=false、原G4 FAIL，NAV-066及物理下游继续阻塞。

## 当前实际完成

在tiger11/tmux复核完整计划、当前依赖、v8末次模型及原162条train/59877步目标；原v5完整train_gate与v8保存结果逐字段一致，六模型及actor未修改，33项v2资源hash匹配。SSH BatchMode因publickey被拒，本次沿用用户已批准的同机tiger11/tmux例外，没有切换主机。

13次误放行分为9次到达分量真值为正、4次仅成本分量应拦截。按预先说明的确定规则，选这13个起点及同地图预算最接近的真实可行起点；对重复配对去重，共25个已见train起点、13组配对。预算较低也可能可行，不假定固定策略的预算可行性单调，不新增轨迹。

这25点中，成本相对MAE仅0.57%，仍有11个成本分量符号错误；到达相对MAE69.86%、Pearson0.3384、9个分量符号错误，联合误放行13次。这是按既有误判选择的诊断子集，不代表全部train/development分布。

只读特征检查显示，到达网络在这25个输入上的增广末层特征，在所报告的float32容差下行秩25；成本为18。说明未发现这些到达特征已完全混在一起，不能据此认定必须换架构。SVD与线性头范数下界只描述固定body的几何；未进行头拟合、不证明有限精度优化成功，也不证明全任务观测或容量足够。

16项基本检查通过：真实标签配对与去重、非单调例、原共同停止标签不变、小成本误差仍需拒绝、合成优化步/actor不变、非有限停止、未批准入口阻断、局部/全train即使通过也不得开发评分或部署、禁止扩候选与覆盖输出。合成小网络测试不使用冻结候选，不是正式训练。真实候选优化/新回合/开发评分均0。

机器证据见[只读诊断](../reports/nav065_local_probe_preflight_v9.json)及[准备复核](../reports/nav065_local_probe_v9_preparation_review.json)。全部旧失败、记录、模型与报告保留。

## 下一次有界诊断提案，尚未批准

参考论文《Solving Minimum-Cost Reach Avoid using Reinforcement Learning》§4.2先固定确定性策略并重新估值，再做预算根搜索；RC-PPO/rl/root_finding.py依靠联合价值的正负。本轮仍在验证估值能力，不提前启动搜索。

[隔离入口](../scripts/probe_nav065_local.py)已接通v8末次来源、原共同停止oracle、确定配对、原数值指标、原v5全train检查及CPU重载。拟在一份深拷贝上，用这25个起点的原cost/reach数值标签做全批次回归，正/非正reach分层等权；原网络结构和两critic保留，不改actor、预算输入、W、beta=0或最终门槛。

| 项目 | 拟申请范围 |
| --- | --- |
| 来源 | 固定full_s0的v8末次模型，不挑中间checkpoint |
| 拟合数据 | 已固定的25个train起点，不使用development |
| 优化 | 一次新Adam，lr=1e-3，最多400步，CPU单线程 |
| 停止 | 首次局部数值与全部分量/联合符号同时通过即停；否则到上限留失败 |
| 新采样/开发评分 | 0 / 0 |
| 产物 | 新独立目录中的diagnostic-only模型与逐步日志，禁止部署或自动扩展 |

局部数值阈值直接沿用原相对MAE≤35%、适用时Pearson≥0.70，并要求所选全部分量和联合符号正确；这些仅是小拟合诊断标准，不是替换原v5 train/G4的验收。拟合前后都直接调用原完整162回合train_gate，单列对其造成的所有退化，局部甚至全train通过也不运行development或发布部署包。

lr=1e-3与原Phase II重拟合设置一致，但相对v8同时改变了样本范围、批次和优化器状态；这是检验局部可拟合性的尝试，不能把结果归因于某个单独因素，也不是新旧方法公平性能比较。最多400步的上限只覆盖本次诊断，不保证通过。

若局部能拟合，下一步优先研究全量训练中的目标平衡/优化干扰；若不能，先分析各分量曲线、精度和表示，仍不能仅凭一次失败断言架构不可能。分支网络、残差参数化等是待证据评审的后续候选，本次未实现或授权训练它们。

[提案](../config/nav065_local_probe_proposal_v9.json)approved=false；没有新的批准登记或正式run。原v8单次尝试已结束，不复用其未执行边界额度。新数字额度须按主计划uncertainty_policy明确确认后，冻结源码/配置hash和批准pin，再进行这一项诊断。不得自动扩六候选、调低标准或将小样本记忆当成泛化。

## 执行位置与交付

host=tiger11.ios.ac.cn；tmux `nav064-r14-repair-20261115:nav065-v9-preparation`；cwd `/home/caohy/repositories/Paper/CaMRA/navigation`；解释器 `/pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python`（3.11.15）。

已执行：`<Python> -B -m pytest -q tests/test_nav065_local_probe.py`；`<Python> -B -u scripts/probe_nav065_local.py --diagnose reports/nav065_local_probe_preflight_v9.json`。基础日志及环境证据位于`.worklog-testing/NAV-065/local-v9-preparation/`；正式诊断入口需要独立批准文件，不把示例命令当成已执行。

固定v2、校准、原目标、观测/架构、论文及RC-PPO参考均保留。NAV-065只有六候选完整train、原development/G4及部署身份全部满足后才能完成，当前准备成功不解除任何下游阻塞。

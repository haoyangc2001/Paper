# NAV-065 保存轨迹离线重拟合 v2 结果

用户于2026-11-18确认六候选各最多300次新增critic更新、总上限1800、不新增轨迹；六组train均通过后才能统一复验旧development一次。按用户要求由主agent验证，未安排专用测试子agent，不称独立测试。批准登记为[prereg v2](../config/nav065_saved_refit_prereg_v2.json)，pin为`a6fe42eb12e94fa857af86702a8b617d2a93d02561e763620d887178afc4705f`。

本轮已结束并停在`HELD_TRAIN_GATE`。实际640次更新，复用540条train轨迹、207298步；新增采样0、development重评分0。没有生成新G4或部署包，原G4 FAIL仍有效。NAV-065=false，计划52/67，NAV-066及下游未启动。四个train通过不能写成四个部署候选通过。

| 候选 | 新增更新 | train检查 | 起点假可行 | 主预算实际可行/总数 |
| --- | ---: | --- | ---: | ---: |
| 无校准 seed0 | 10 | PASS | 0 | 18/18 |
| 无校准 seed1 | 10 | PASS | 0 | 18/18 |
| 无校准 seed2 | 10 | PASS | 0 | 18/18 |
| 完整方法 seed0 | 10 | PASS | 0 | 18/18 |
| 完整方法 seed1 | 300 | FAIL | 2 | 18/18 |
| 完整方法 seed2 | 300 | FAIL | 2 | 17/18 |

## 结果与失败原因

同一540条train的起点假可行13→4，假不可行0→2；成本起点MAE 1.585→1.994 J、沿途成本MAE 1.343→1.431 J。可见边界误判有改善，平均误差并非所有项都改善。六组成本/reach平均误差及相关性均满足原标准，但零假可行门槛仍未过。这里的540是五档预算、六候选的完整诊断分母，不是540条主预算成功率。

| 失败候选 | 地图 | k | 真实事件 | 最终超支 | 起点reach真值→预测 |
| --- | --- | ---: | --- | ---: | --- |
| full seed1 | l1_empty-s1001 | 1.00 | 安全到达但超预算 | 3.064 J | 0.00332→−0.26057 |
| full seed1 | l1_empty-s1004 | 1.00 | 安全到达但超预算 | 2.875 J | 0.00301→−2.00094 |
| full seed2 | l1_empty-s1001 | 1.00 | 安全到达但超预算 | 1.304 J | 0.00150→−1.49459 |
| full seed2 | l2_single_detour-s1006 | 1.15 | 全程安全、预算内，但超时未到达 | 0 J | 0.00517→−0.54046 |

前三例共同停止成本被低估2.36–3.11 J，reach的小正数也被误判为负数，导致W的两侧同时错误放行。共同停止可早于最终到达，因此共同停止成本差和最终超支不同，不能混作一项。

最后一例原始train轨迹在45秒结束时距目标中心0.577576 m，目标半径0.5 m，尚差约7.76厘米；仍剩31.103 J。其真实失败是未进入目标区，不是预算不足。actor固定后，critic重拟合不会改变这条轨迹，而reach预测仍错误为负。

## 本轮方案的门槛适用性缺口

批准方案将主预算train门槛设为18/18，却没有在开跑前核对固定actor在train主档是否确有18条可行轨迹。这是方案检查遗漏：full seed2真实可行只有17/18；即使评价网络完全正确，也至多正确接受17条，无法同时达到18条正确接受和零假可行。

该缺口不能掩盖仍存在的四次误判，也不能在本轮改标准后回填PASS。批准门槛、失败日志和权重原样保留。NAV-064原验收针对development主预算，原36/36结果没有被本次不同train地图上的失败推翻，不能混用分母。

后续应先单独评审train门槛的适用性，把策略真实可行率与评价网络是否正确判断可行/不可行分开；原development标准和零假可行标准继续保留。进一步拟合方案需针对小正reach及预算边界保号提出可检验方案，并在取得所需明确授权后执行。本轮不自动改门槛、追加更新或重训actor。

## 执行、复算与保留

实际执行在`tiger11.ios.ac.cn`，tmux会话`nav064-r14-repair-20261115`的`nav065-saved-v2`窗口；cwd为`/home/caohy/repositories/Paper/CaMRA/navigation`，解释器为`/pub/data/caohy/miniconda/envs/env_isaaclab_v0472/bin/python`。15:00:49启动，15:01:35结束，耗时46.10秒，exit0仅表示程序按规则正常停止。

原始输出：`/pub/data/caohy/work/camra_navigation/nav065_saved_refit_v2/`。启动/资源/输入前后hash及退出证据：`/pub/data/caohy/work/camra_navigation/nav065_saved_refit_launch_20261118_v1/`。

主agent在同机tmux窗口`nav065-saved-review`逐模型CPU重载、逐tensor核验actor一致、两critic更新、批准pin和原输入不变，核验更新序号及每10次train检查/首次通过即停。复用原始轨迹重建全部540条真值和预测，并以独立的暴力后缀argmin、J账本和事件记录交叉核验共同停止和可行性。未评分development；33项v2身份沿原生产预检核验。两个失败候选各300上限已耗尽，不转移其他候选未用额度。

完整[复算报告](../reports/nav065_saved_refit_v2_review.json)、[逐起点结果](../../../.worklog-testing/NAV-065/saved-refit-v2-review/train_cases.json)、[审计脚本](../../../.worklog-testing/NAV-065/saved-refit-v2-review/audit.py)及[执行日志](../../../.worklog-testing/NAV-065/saved-refit-v2-review/audit.log)可追溯全部分母。训练生产源码、本轮登记、actor、v2、旧模型、论文与只读参考未修改；本轮只新增批准登记、独立输出、审阅及追加记录。

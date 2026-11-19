NAV-067已通过工程接入验收，计划55/67，下一项NAV-043尚未执行。固定v2、新版两候选、45秒/obs-v7和RC-PPO原理保持，原科学FAIL保留。

六项实际受控检查通过：中间碰撞、危险历史、转向扫掠、同刻事件、异步结束和原生接触。前五项为人为位姿写入后的实际读回，第六项是初次放置后的原生接触响应；均不计作自然导航成功。

原六条pilot和本次六条重复均完成命令真实观测、10/50/200Hz、复位、终止、部分周期及机械功复算。每种配置两次，只报告下表差异；不宣称未知场景稳定或物理逐位一致。

| 地图 | 方法 | 原/重复结束 | 时间差(s) | 机械功差(J) | 最大位置差(m) |
| --- | --- | --- | ---: | ---: | ---: |
| L1_empty | uncalibrated | timeout/timeout | 0 | 0 | 0 |
| L1_empty | full | goal_reached/goal_reached | 0 | 0 | 0 |
| L2_single_detour | uncalibrated | timeout/timeout | 0 | 0 | 0 |
| L2_single_detour | full | timeout/timeout | 0 | 0 | 0 |
| L3_multi_path | uncalibrated | timeout/timeout | 0 | 0 | 0 |
| L3_multi_path | full | goal_reached/goal_reached | 0 | 0 | 0 |

这12条是共同预算接线记录。到达、超时、安全历史和超预算逐条保留在报告中；工程验收不代表预算内导航性能通过，也不替NAV-043/G5放行。

旧NAV-042的clean声明修正见独立amendment，本次补充重复已完成；旧报告/失败/模型均不覆盖，旧新差异不能全部归因于关闭扰动。

本次恢复累计启动10次应用（含原失败），单GPU墙钟6425.419秒，环境子步预留57830，未超10应用/4小时/60000上限；预留不等于实际消耗。

后续应先检查实际部署策略的跟踪、45秒时限、能量前缀和原裕度适用性。当前恢复授权不含NAV-043新执行，不自动增加物理回合、训练或预算查询。

[完整验收和结果](../reports/nav067_engineering_acceptance_v1.json)；原始大轨迹保留在登记实验目录，Git保存校验索引。

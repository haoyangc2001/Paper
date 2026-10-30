# 方法契约（TASK-003）

本文件用于内部模拟终稿的实现说明设计；代码核对不代表 Go2 已实现这些设置。

## 已核实的来源

- 原文 `CORL/references/Solving Minimum-Cost Reach Avoid using Reinforcement Learning.pdf`：第 3 页式(1)为确定性动力学；第 4–5 页式(7)–(13)用历史安全标记和剩余预算建立增广可达问题；第 6 页采用折扣 reach 目标与 PPO 改造；第 7 页先确定化策略、再评价价值、最后搜索预算。原文第 1 页明确成本也可以表示平滑性，不能声称理论绝对不能控制步态。
- `RC-PPO/src/model/actorcritic.py`：独立 actor / Value_Network，256–256 tanh；actor 为对角高斯、状态无关 log standard deviation。
- `RC-PPO/src/rl/EC-EFPPO.py`：实际实例化 policy、energy、h 三网络；合成残差 `max(V_reach,V_energy-energy)`；成本与 reach 分别训练。
- `RC-PPO/src/rl/gae.py`：`calculate_indexs3`选最优停止时刻；`calculate_gae2`累计成本到该时刻；`calculate_gae_reach4`计算 reach 的多步 lambda 返回。
- `RC-PPO/src/rl/EFPPO_utils.py`：`_ecefppo_update`最小化 `max(r*A,clip(r)*A)-entropy`；两个 critic 各自使用 clipped squared value loss。这里 A=target-value，低值更好，不可直接照抄最大化奖励的 PPO 符号。
- `RC-PPO/src/rl/utils.py`：Adam eps=1e-5，global gradient clipping；`arguments.py`给出原代码默认值，不是 Go2 已测配置。
- `RC-PPO/src/rl/root_finding.py`：双critic合成判据的一维二分法。当前稿改用实测指标阈值的有限预算网格，这只是工程选择规则，不是连续最优证明。

## 三网络与两裕度

| 组件 | 输入/输出与职责 | 裕度作用 |
|---|---|---|
| actor | 局部观测、目标、历史安全标记、剩余预算 → 3维速度命令 | 经由更新后的预算、危险集与两个评价目标间接影响策略 |
| cost critic | 同一增广输入 → 所选停止时间前剩余成本 | alpha 改正 running cost 及其返回目标 |
| reach critic | 同一增广输入 → 目标与历史安全的 reach residual | delta 通过额外障碍膨胀改变 unsafe indicator 和目标 |
| 合成残差 | max(V_h,V_c-z)，统一无量纲后比较 | 用于 actor 的成本式优势；不是两个独立最优值的精确代数等价 |
| frozen tracker | 本体状态、速度命令 → 12个关节位置目标 | 不接收 alpha/delta，不在导航训练期间更新 |

两个 critic 的目标共享同一轨迹和停止选择。不能把两种独立期望的最大值宣称为理论单critic Bellman 固定点，也不能让 cost critic 优化零时间停止而 reach critic 优化另外一个时刻。正文给出理论概述；附录明确采用双critic近似和共同停止目标。

## 需要实施的修订

1. 方法的 Phase I 展开双critic、共享停止和 actor 的合成目标；总览图现有两个评价网络保留。
2. 附录以成本式 PPO 最小化统一符号，给出多步 reach 返回与成本返回、预算归一化、训练终止规则。
3. alpha 修正名义计费；真实执行 cost 单独累积用于 BVR，不能把内部预算耗尽等同于实测超预算。
4. delta 是物理 footprint 外的额外距离；CR/MC 始终用物理几何，不能用各方法不同膨胀障碍来统计。
5. 固定底层缓解扰动并隔离运动稳定任务，但不会消除随机性；理论并未排除把姿态危险或平滑成本加入模型。分层理由是工程训练难度与校准一致性。

## 验收

原PDF正文及上述代码已核对；单critic理论和双critic实现的差异有定位；没有更改原理论、训练源代码或声明新增安全定理。后续 TASK-008 / TASK-020 落实为完整实现约定。

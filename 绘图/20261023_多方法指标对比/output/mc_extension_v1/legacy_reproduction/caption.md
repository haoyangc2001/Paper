# 配套图注

## 当前假设数据预览用图注

Scenario-level navigation performance of seven methods. Each row is a method; circles, triangles, and squares denote moderate clutter, narrow passages, and dense obstacles. Panels report success rate (SR), collision rate (CR), budget violation rate (BVR), and execution cost. Blue denotes HD-MCRA, orange RC-PPO, and rose RA. The PPO variants use light and dark purple, and the CPPO variants use light and dark green, with the lighter shade indicating beta L and the darker shade beta H. All point estimates are assumed data. Horizontal bars show manually specified hypothetical estimate ranges for layout review; they are not measured confidence intervals, standard deviations, or standard errors. Cost retains the provisional units of the draft.

七种方法在三个导航场景下的整体表现。每行对应一种方法，圆形、三角形、方形分别对应普通障碍、窄通道和密集障碍。四列展示成功率、碰撞率、预算违规率和执行成本。HD-MCRA为蓝色、RC-PPO为橙色、RA为玫红色；PPO两种设置使用深浅紫色，CPPO两种设置使用深浅绿色，浅色对应beta L、深色对应beta H。所有点估计均为假设数据；横向短线为逐点指定的假设估计范围，用于版式审阅，不是实测置信区间、标准差或标准误。成本沿用草稿单位。

## 使用边界

- 不能根据本图声称显著优于基线、获得某个覆盖概率、满足真实预算或已验证实机安全。
- 区间覆盖的是每个汇总估计的示意范围，不是一次运行的取值范围。各指标区间不构成联合分布。
- 不同场景的三个点不是重复实验样本，不用它们计算误差线。
- 成本必须与到达情况一起解释；在成本定义与失败计费未明确前，不将较低均值单独等同更高效率。
- 正式替换时，全部中心与区间应来自同一明确统计协议下的真实记录，重新确定区间类型、统计单位和置信水平，并重写此图注。

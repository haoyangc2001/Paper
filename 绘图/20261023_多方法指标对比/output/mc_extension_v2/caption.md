# 配套图注

## 当前五指标版（假设数据）

Scenario-level navigation performance at the common reference budget. Panels report SR, execution cost, CR, BVR, and mean per-trial minimum physical clearance (MC), in the same order as Table I. Marker shapes distinguish the three scenarios; colors distinguish methods. All centers and horizontal ranges are assumed data; the ranges are not measured standard deviations or confidence intervals. Table I reports equal-scene means. Cost uses provisional draft units; MC is in metres and averages all trials through goal, collision, or timeout.

共同参考预算下的场景级导航表现。五列依次为SR、执行成本、CR、BVR和平均最小物理间距MC，与表一顺序一致。点形区分普通障碍、窄通道、密集障碍，颜色区分方法。中心值及横向区间均为假设数据；区间不是实测标准差或置信区间。表一报告三场景等权平均。成本沿用草稿单位；MC为全部试验截至到达、碰撞或超时的全程最小物理机身间距的均值，单位为米。

## 使用和替换

唯一点数据入口为本次--data-dir指定目录中的assumed_intervals.jsonl；CSV和本说明均为派生副本。原84条记录不变，新增MC001—021的中心来自旧生成脚本，42个端点逐点人工指定；不生成实验样本。区间不能从三个场景差异计算，也不能平均端点得到总体区间。各指标为边际估计，不构成联合分布；MC均值不等同碰撞概率或delta的安全保证。

替换真实实验时应使用同一统计口径的真实中心及真实区间，并填写protocol中的来源、样本组织、区间估计方法和对应图注；不能仅改hypothetical标签。成本仍须结合成功率及失败计费解释。

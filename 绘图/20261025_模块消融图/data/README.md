# 最终绘图数据入口

唯一数值输入为reviewed_points.jsonl：25行，16个main点进入四面板，9个support点供关联审查。每条包含value、单位、来源、原值及审查版本。baseline_points.json保留原值，禁止改写历史。plot_protocol.json规定指标/行顺序和数据性质。

全部中心值是假设数据。完整方法/RC-PPO采用既有三个场景的等权平均全精度，论文一位/两位显示值保持一致。无消融误差区间，细连接不是误差线。

改数值：编辑目标point_id的value，在reviews/point_review.jsonl追加独立修订记录，并核对所有相关点。10个跨图锚点不能在本项目孤立改动。所有来源更新一致后再更新协议中的锚点并运行检查。不要修改源工作簿的副本而期待本脚本读取它。

改外观在config/style_horizontal.json；改中心后重跑校验和绘图。真实数据替换前需落实试验数、seed/地图分组、成本与预算、物理碰撞定义和区间估计方式，不能只把hypothetical改名便宣称实测。

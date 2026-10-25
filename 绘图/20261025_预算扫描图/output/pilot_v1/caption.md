# 预算扫描图配套图注

## English caption

Budget-response curves (assumed data), averaged equally across three scenarios.
All panels use the normalized initial budget, z0/zref. The reference at 1.00 matches
the main comparison table. Dashed horizontal limits mark SR >= 60%, BVR <= 15%,
and CR <= 20%. Outlined points in the BVR panel identify the smallest tested budgets
meeting all three criteria (HD-MCRA: 0.90, RC-PPO: 0.96, RA: 1.12). RA reuses the same trajectories at every
budget, so only its BVR changes. Execution cost averages all trials through goal,
physical collision, or timeout; budget violation does not terminate evaluation.
The SR and cost axes have nonzero lower bounds. Lines connect the given candidate
values as visual guides; no uncertainty intervals or continuous optimal budgets
are inferred. These manually specified values are for manuscript layout review,
not measured experimental evidence.

## 中文说明

三场景等权均值的预算响应（假设数据）。横轴是初始预算与场景参考预算之比；
1.00的均值与主表对应。三个水平阈值分别为SR≥60%、BVR≤15%、CR≤20%。
BVR面板的同色空心强调圈对应同时满足三项要求的最小已测试候选；
不代表连续最优预算、每个场景分别达标或有统计可靠性保证。

RA复用相同轨迹，所以SR、CR、Cost保持水平，仅BVR随阈值改变。
成本包括所有回合到达、物理碰撞或超时前的消耗；超预算不提前终止。
SR/Cost纵轴没有从零起算，刻度明确显示。折线只连接已有点，未做平滑、
未增加噪声或误差带。图面可用于正式版式，分享时应同时保留本图注的数据性质。

当前0.90/0.96/1.12的选择由输入数据重新计算；首次BVR通过余量仅为
0.3/0.1/0.6个百分点，不应表述为稳健达标。0.96至1.00的成功率变化保留
原假设，原始日志验证仍未完成。输入与逐点理由均在本项目data及reviews中。

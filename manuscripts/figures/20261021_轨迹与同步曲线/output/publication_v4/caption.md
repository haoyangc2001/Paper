# 轨迹与同步曲线：配套图注

## 当前版本说明

当前数据为逐点编写的假设数据，用于审阅版式与论证关系，不是实测、策略输出或物理仿真结果。图面采用论文排版；数据性质保留在本图注、输入配置、README、生成记录和 PDF 元数据中。

## 中文图注

同场景轨迹与同步诊断曲线（当前为假设数据）。蓝色为 HD-MCRA，橙色为 RC-PPO，两者采用相同的冻结低层速度接口、起终点、机器人足迹和初始预算 85。
(a) 实线表示执行轨迹，浅虚线表示相同命令理想跟踪下的名义轨迹。深灰为实体障碍，浅灰为计入机器人足迹的扩张区域，灰色点线在其外增加 δ=0.12 m 的规划余量。局部图显示各方法经过 A 时的机器人足迹。
(b) 执行机身与实体障碍之间的间隙，零线表示接触边界。
(c) 执行与名义累计成本、共同预算，以及 HD-MCRA 的校正名义计费 (1+α)J_nom，其中 α=0.18。
(d) 线速度跟踪误差的欧氏范数。A/B 对应各自首次采样越过指定位置的时刻（A: x = 1.9 m; B: x = 3.3 m），不表示同时到达；各曲线在本方法首次终止时结束，跟踪误差绘至最后一个有动作记录的区间起点。

## English caption

Assumed-data illustration; the records are manually authored, not measurements, policy outputs, or physical-simulation results. Same-scene trajectories and synchronized diagnostics for HD-MCRA (blue) and RC-PPO (orange), using a shared frozen low-level velocity interface, start, goal, footprint, and initial budget of 85. (a) Solid lines: executed trajectories; faint dashed lines: nominal trajectories under ideal tracking of the same commands. Dark gray: physical obstacles; pale gray: footprint-expanded regions; dotted boundaries: an additional planning margin δ=0.12 m. The inset shows robot footprints at each method's event A. (b) Executed body-to-obstacle clearance; zero indicates contact. (c) Executed and nominal cumulative costs, the common budget, and HD-MCRA's calibrated nominal charge (1+α)J_nom, with α=0.18. (d) Euclidean linear-velocity tracking error. A/B mark each method's own first sampled crossing (A: x = 1.9 m; B: x = 3.3 m), rather than simultaneous arrivals. Each trace stops at its own first termination; tracking errors are available only at action-interval start times.

## 实验设置与结果解释

- 当前成本单位：illustrative cost units。图中 a.u. 表示示例单位；成本定义与论文主表单位尚未对齐，不能直接等同。成本函数和权重见输入 scenario.json 及 data/versions/v2/protocol.md。
- 当前事件时刻：HD-MCRA：A=6.8 s、B=11.2 s；RC-PPO：A=6 s、B=10 s。
- 本案例两方法均未超过预算，不能用其证明 α 降低预算违约率。本图保留安全间隙、耗时和成本之间的取舍，不单独支持总体性能结论。
- 后续在新数据目录替换真实日志并重新计算；同时更新输入的数据来源、成本定义与配置。图注中的比较条件和论证边界也应随真实实验核对。

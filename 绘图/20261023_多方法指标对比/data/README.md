# 多方法指标对比图：最终绘图数据

整理日期：2026-10-25。此项目本来已采用项目级 `data/`；本轮保留原输入，并补入当前输出数值表的原样副本。

- `assumed_intervals.jsonl`：实际绘图输入，84个点；每点包含方法、场景、指标、中心值、左右区间端点与理由。修改数据以此文件为准。
- `protocol.json`：方法与场景编码、指标单位、颜色、版式、数据性质及原中心值核查来源。
- `assumed_point_estimates.csv`：从 `../output/assumed_point_estimates.csv` 按字节复制的最终数值导出表，便于查阅；不是第二个独立输入源。修改JSONL后应重新导出，避免手工维护两份值。

当前成图：`../output/metrics_comparison_assumed.pdf`。脚本为 `../scripts/plot_metrics_comparison.py`，输入仍是本目录的JSONL与协议配置，中心值校验还会参考协议指定的原数据生成脚本。

这里的区间是手动指定的假设估计范围，不是实测标准差或置信区间。预算扫描图的不同预算点没有这些区间，不能把本图区间直接移植过去。本轮没有改84个中心值或区间端点。

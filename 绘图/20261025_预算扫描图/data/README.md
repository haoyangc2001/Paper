# 预算扫描图：最终数据副本

整理日期：2026-10-25。两个文件均从当前预算图的来源原样复制，并核对SHA256一致。

## 文件与来源

1. `experiment_results_rcppo_ra_v3.xlsx`：复制自仓库 `CORL/data/experiment_results_rcppo_ra_v3.xlsx`。为完整保留来源关系，保留原工作簿全部工作表，不删改其中其他结果。
2. `figure_sources_snapshot.json`：复制自 `CCDC/output/1020/consistency_review/figure_sources.json`。其中 `budget_data` 是当前预算图使用的96个中心值；`budget_workbook_sha256`、`budget_figure_sha256` 分别记录原工作簿和已生成图片的哈希。另含同一轮其他图的来源字段，不属于额外预算数据。

## Excel里看哪些表

| 工作表 | 用途 |
| --- | --- |
| `budget_metric_wide` | 原绘图函数实际读取的宽表：3种方法×4项指标，每行8档预算，共96个中心值 |
| `budget_long_format` | 同一批数据的逐条展开形式，方便筛选核对 |
| `minimum_feasible_budget` | 同时满足SR、CR、BVR条件的最小已测试候选预算及对应指标 |
| `metadata` | 数据性质、统计口径、待核实信息 |

八档预算比例为0.80、0.85、0.90、0.96、1.00、1.05、1.12、1.20；三种方法为HD-MCRA、RC-PPO、RA；四项指标为SR、CR、BVR、Average Cost。

比例值以0–1存储，例如0.60绘成60%；成本沿用草稿单位。参考预算1.00与主表等场景均值对齐。当前最小通过测试预算为HD-MCRA 0.90、RC-PPO 0.96、RA 1.12，条件为SR≥60%、CR≤20%、BVR≤15%。

## 后续如何读取和修改

现有 `CORL/scripts/plot_experiment_results_v3.py` 中的 `build_budget_data(path)` 可显式传入本目录的工作簿路径；`plot_budget_grid(data, output_dir)` 接收读取后的数据。后续新绘图脚本应以本项目 `data/` 为输入入口，不再依赖跨目录的CORL工作簿。

本次只是复制，旧的CCDC审计/重绘脚本仍指向原CORL工作簿，没有自动切换。不要修改新副本后又运行旧入口而误以为图片已经更新。旧数组定义还保留在 `CORL/scripts/generate_experiment_results_v3.py`，不应在未核查时重新生成覆盖本地副本。

`figure_sources_snapshot.json`是当前成图输入的归档快照，不会自动跟随工作簿变化。修改后应按同一来源重新导出、重绘与核对，不要把旧快照当作最新结果。

数据性质仍为之前统一的假设数据；只有中心值，没有逐预算的真实区间或逐试验日志。RA复用相同轨迹，因此SR、CR、成本固定，只有BVR随阈值变化。

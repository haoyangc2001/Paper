# 五指标版绘图数据

版本：assumed-mc-extension-v1，2026-10-25。所有中心与区间均为假设数据。

- `assumed_intervals.jsonl`：唯一新版点数据源，共105条；P001—P084逐字段沿用旧版，MC001—MC021为新增21个MC点，每点独立指定左右端点及理由。
- `protocol.json`：五列顺序SR/Cost/CR/BVR/MC、方法颜色、场景点形、米单位及显示精度、89×70mm单栏排版。
- `assumed_point_estimates.csv`：从JSONL导出的查阅副本，含假设性质及端点；修改应回到JSONL。

MC中心来自`CORL/scripts/generate_experiment_results_v3.py`的`MAIN_DATA["Minimum Clearance"]`，与原工作簿21值及主表7均值核对一致。MC为全部试验各自全程最小物理机身间距的均值，单位m，统计至到达、碰撞或超时。它不使用膨胀后的规划边界，也不等于空间裕度δ。

42个新增端点是逐点人工假设估计范围；不代表实测标准差、标准误、95%CI、逐回合距离范围或已验证覆盖率。每点参考同场景其他指标、方法和此前相关点，三个场景完成后分别审查，记录位于`../../reviews/mc_extension_v1.md`。不从三个场景差异算误差，不平均端点生成总体区间。

重绘命令见[项目README](../../README.md)。当前输出位于`../../output/mc_extension_v1/`。修改后重绘，并同步查阅CSV：

```powershell
Copy-Item -LiteralPath '绘图/20261023_多方法指标对比/output/mc_extension_v1/assumed_point_estimates.csv' -Destination '绘图/20261023_多方法指标对比/data/mc_extension_v1/assumed_point_estimates.csv'
```

后续真实数据应另存新版本、保留本假设快照；从明确的逐试验记录计算中心与适当统计区间，更新真实来源、样本组织、定义和图注。旧假设中心核查器只适用于本轮，不要求真实结果匹配当前数值。对应论文中的表一、图二及相关汇总应一起更新。

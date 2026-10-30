# 五指标宽版数据与排版

当前版本：assumed-mc-extension-v2-wide。按用户要求，图幅从第一版的89×70mm调整为220×90mm；原四列图为181×90mm，因此高度与旧图严格相同，宽度增加约22%。字体8pt、刻度7.5pt、标记3.8pt恢复到旧四列风格，保留五列SR/Cost/CR/BVR/MC。

`assumed_intervals.jsonl`为本版唯一点数据源，与mc_extension_v1中的105条记录字节一致，中心及210个端点均未修改，仍全部为假设数据。MC中心及逐点端点理由保持原定义；本次仅改protocol.json中的版式、标题和刻度。`assumed_point_estimates.csv`为导出查阅副本。

成图在`../../output/mc_extension_v2/`。旧四列版及89×70mm五列版均保留。论文仍用columnwidth单栏等比例插入，因此源图恢复90mm高不等于论文内按90mm高放置；源图加宽后在同一栏宽下显示为更扁的比例。

从仓库根目录重绘：

```powershell
& 'D:/codesoftware/Miniconda/python.exe' -X utf8 -B '绘图/20261023_多方法指标对比/scripts/plot_metrics_comparison.py' --data-dir '绘图/20261023_多方法指标对比/data/mc_extension_v2' --output-dir '绘图/20261023_多方法指标对比/output/mc_extension_v2'
& 'D:/codesoftware/Miniconda/python.exe' -X utf8 -B '绘图/20261023_多方法指标对比/scripts/audit_mc_extension.py'
```

改数据应改JSONL并重新导出CSV，改外观应改protocol.json。真实实验替换的口径及21个MC逐点理由见[第一版数据说明](../mc_extension_v1/README.md)与项目审查记录；不得将假设区间当成实测CI、SD或SE。

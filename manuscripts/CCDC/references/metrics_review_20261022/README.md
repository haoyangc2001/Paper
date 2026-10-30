# 结果对比图调研（2026-10-22）

本轮针对 CCDC 1020 稿的综合指标图，检索其他论文如何组织多方法、多场景、多指标对比，并提出重绘建议。未修改论文正文、现有结果图或实验数据。

- [详细分析与重绘建议](结果对比图调研与重绘建议.md)
- [19 组原图的 PDF 阅读册](metrics_reference_atlas.pdf)：每页一组原图，附来源、原文页码、图号和借鉴要点；原图仅作内部阅读与设计参考。
- [原图速览一](figures/case_contact_01.png)与[原图速览二](figures/case_contact_02.png)
- [裁图坐标及出处](figure_cases.json)
- [原文来源、版本、SHA256 和图注](source_manifest.json)
- [检索记录](search_log.json)：6 组 arXiv 检索，保留 99 条候选记录、95 个不同 URL；其中有不相关候选，不等于详细阅读了 95 篇。
- [当前图和主表核对](current_figure_audit.json)：包括约 3.3 pt 的图例字号，以及 RA 的 BVR 在旧生成脚本与主表间存在的差异。

本轮筛查 21 篇完整 PDF 的相关图注和实验页，重点截取其中 16 篇的 19 组图／表。新增下载 10 篇公开原文，其余 11 篇使用仓库已有副本，不再重复复制。具体读图版本以本地 PDF 为准；例如 SEA-Nav 的案例来自归档 v1，在线已存在 v2，两者不混称。

目录内容：

```text
papers/                     本轮新增下载的 10 篇原文
text/                       21 篇原文的带页码文本，便于检索图注
previews/                   相关完整原文页与阅读册预览
figures/                    19 组原图截取及两张速览
figure_cases.json           原图页码、裁图范围和借鉴说明
source_manifest.json        原文 URL、读取版本和文件哈希
search_log.json             检索结果
current_figure_audit.json   当前图的版式及假设数值一致性核对
metrics_reference_atlas.pdf 原图阅读册
```

复查脚本为 `../../scripts/review_metrics_references_20261022.py`。在仓库根目录使用 `D:/codesoftware/Miniconda/python.exe` 运行：

```powershell
& 'D:/codesoftware/Miniconda/python.exe' './CCDC/scripts/review_metrics_references_20261022.py' scan
& 'D:/codesoftware/Miniconda/python.exe' './CCDC/scripts/review_metrics_references_20261022.py' preview
& 'D:/codesoftware/Miniconda/python.exe' './CCDC/scripts/review_metrics_references_20261022.py' render
& 'D:/codesoftware/Miniconda/python.exe' './CCDC/scripts/review_metrics_references_20261022.py' audit
```

`acquire` 会联网重新查询并更新原文元数据；已有 PDF 不会覆盖。`audit` 读取旧生成脚本中已定义的假设数组及主表，不执行其生成入口，也未打开或修改 Excel 工作簿。

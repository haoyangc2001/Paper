# CCDC 1029：无附录投稿版本

- [当前正文](main.tex)
- [当前论文PDF](../../output/1029/main.pdf)
- [参考文献库](references.bib)

当前PDF为7页，含23篇参考文献；图表均为单栏，保留IEEE会议模板。2026-10-30完成参考文献补充、基线与消融说明修订及英文润色；最新写作提交为`6ce65ba`。润色未改变方法、定义、公式和实验数值。

## 编译

在仓库根目录执行：

```powershell
& CCDC/scripts/build-1029.ps1
```

脚本通过`latex-paper-en/scripts/compile.py`调用latexmk，输出到`CCDC/output/1029/`。编译辅助文件和页面PNG均可重建。

## 资料入口

图片、数据和参考材料分别位于`CCDC/images/1029/`、`CCDC/data/1029/`、`CCDC/references/1029/`。来源和推断台账保留，后续需按实际实验日志替换相关示例统计。

本版本由1020稿复制后去除附录建立，历史完整稿保留在`CCDC/latex/1020/`及本目录`archive/`。参考原文、公式与历史定义时可查阅完整稿，不能把历史稿视为最新投稿版本。

作者为Haoyang Cao、Yuhua Guo、Bai Xue；第二作者保留两个单位，共用地点为Beijing, China。作者来源说明位于`CCDC/submission/1029/作者信息来源.md`。

2026-10-31已清理旧版页面预览、阶段审查和计划记录。需要追溯时，可查阅本机[过程文件备份](../../../.local/cleanup_20261031/README.md)，按原路径选择恢复。当前主稿、样式、参考文献、图片、数据和最终PDF均保留。

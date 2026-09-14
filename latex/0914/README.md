# 0820_submitted —— CoRL 2026 投稿版（编号 #1530）

> **状态：源码已丢失。** 本目录为将来重建 LaTeX 源码预留的位置，目前只有一份说明。

## 这一版是什么

投给 CoRL 2026 的正式提交版本（投稿编号 **#1530**，双盲评审），**已被拒稿**。

- 标题：Hierarchical Minimum-Cost Reach-Avoid Learning for Safe Quadruped Navigation
- 作者块：Anonymous Author(s) / Affiliation / Address / email（双盲）
- 篇幅：**11 页**，6 节（含独立的 §5 Limitations、§6 Conclusion）
- 页脚：Submitted to the 10th Conference on Robot Learning (CoRL 2026). Do not distribute.

## 现有材料（都在本仓库内）

| 材料 | 位置 |
| --- | --- |
| 终版 PDF（11 页，权威原件，勿改） | `output/0820_submitted/1530_Hierarchical_Minimum_Cost.pdf` |
| 终版全文文本提取 | `output/0820_submitted/_1530_extracted.txt` |
| 重建起点：旧版可编译源码 | `latex/0710_legacy/conference_101719.tex` |

## 相对 0710_legacy 的差异（重建时要逐项移植）

| 维度 | 0710_legacy（旧版源码） | 0820_submitted（本版，投稿版） |
| --- | --- | --- |
| 标题 | HD-MCRA: Hierarchical Reach-Avoid Navigation for Quadruped Robots | Hierarchical Minimum-Cost Reach-Avoid Learning for Safe Quadruped Navigation |
| 作者块 | IEEE 模板占位 `Given Name Surname` | Anonymous Author(s) 等 |
| 真机实验 | four out of five（4/5） | **26 out of 30**（约 15 m，含第二个场景） |
| 真机图 | Fig. 4、Fig. 5 两张独立图 | 合并为 **Figure 3(a)(b)** |
| 章节结构 | 5 节，无独立 Limitations | 6 节，**新增 §5 Limitations** |
| 参考文献 | 约 25 条 | **35 条**（新增 SEA-Nav 2026、ProNav、CVPO、DRO、chance-constrained 等） |
| 页脚 | 无 | CoRL 2026 投稿声明 |
| 篇幅 | 6 页 | 11 页 |

## 重建注意事项

1. 从 `latex/0710_legacy/` 复制 `conference_101719.tex`、`IEEEtran.cls`、`references.bib` 到本目录后再改（`.cls` 与 `.bib` 必须和主 tex 同目录，否则 xelatex/bibtex 找不到）。
2. 图片引用路径用 `../../images/...`（本目录与 `0710_legacy` 同层级）。
3. 命令行编译（在仓库根目录执行）：
   ```
   cd latex/0820_submitted
   latexmk -xelatex -interaction=nonstopmode -file-line-error -outdir=<输出目录> conference_101719.tex
   ```
4. 新增的 10 条文献先转成 BibTeX 条目，追加到 `references.bib`，条目信息可从 `output/0820_submitted/_1530_extracted.txt` 的 References 段落抄录。
5. 重建期间若要使用 VS Code 构建，需把 `.vscode/settings.json` 里的 `latex-workshop.latex.outDir` 从 `output/0710_legacy` 改到 `output/0820_submitted`（该配置一次只能指向一个版本）。

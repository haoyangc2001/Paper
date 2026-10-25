# 四足机器人导航论文工作区

## 常用入口

- 当前 CCDC 稿件：[正文](CCDC/latex/1020/main.tex)、[编译 PDF](CCDC/output/1020/main.pdf)、[工作区说明](CCDC/README.md)。
- CoRL 原投稿与后续改稿：`CORL/latex/0820_submitted/`、`CORL/latex/0914/`；审稿材料在 `CORL/submission/`。
- 绘图工程、数据和生成脚本：[绘图目录说明](绘图/README.md)。
- 后续计划与工作记录：`plan/`，以及各会议目录中的 `plan/`。
- RC-PPO 参考实现：`RC-PPO/`；原始论文和参考文献位于 `CORL/references/`。

## 根目录整理（2026-10-25）

根目录中三份内容相同的 PDF 实际是 Zhou 与 Zhang 的参考论文，并非本项目最新稿。现保留一个正式资料入口：

- [2025 年机器人避障参考论文及提取文本](CORL/references/related_navigation_2025/README.md)。
- [阅读导出文本说明](CORL/references/reading_exports/README.md)。

授权截图、重复下载副本和旧的临时排版检查目录已移入本机 `.local/cleanup_20261025/`，可按其中的 `README.md` 恢复。该目录由 Git 忽略，不作为论文资料发布。正文、图表、实验数据、计划及 RC-PPO 源码保持原位。

临时目录 `_tmp_build_consistency/` 仍可能由排版核查脚本自动创建；它是可重建的检查产物，已在 Git 忽略规则中。

# CCDC 1020 论文版本

建立日期：2026-10-20。

## 内容与模板来源

- 最新正文来源：`../../../CORL/latex/0914/main.tex`（源码修改日期 2026-10-05）。
- 文献数据库：复制同版 `references.bib`，条目内容不改动。
- 官方模板来源：`../template/Conference-LaTeX-template_10-17-19/`。
- 论文编译入口：`main.tex`。
- `IEEEtran.cls`、`conference_101719.tex`、官方示例 PDF、`IEEEtran_HOWTO.pdf` 和 `fig1.png` 均原样复制。`conference_101719.tex` 是官方格式示例，不是论文入口。
- 论文使用的三张图片原样复制到 `../../images/1020/`，不依赖 CORL 目录编译。

## 本次迁移边界

保留原稿标题、摘要、章节、论述、公式含义、实验数值、附录和参考文献数据库；不进行篇幅压缩、研究内容改写或数据替换。原 CORL 0914 稿和 CCDC 原始模板不改动。

只进行模板适配：

1. 使用官方 `\documentclass[conference]{IEEEtran}`，保持默认 10pt、US Letter 和双栏。
2. 使用官方作者块、摘要及 `IEEEkeywords` 环境；保留来源稿的作者占位内容，不据此断言 CCDC 要求匿名。
3. 将 CoRL 的 `\citep` 改为 IEEE 数字引用 `\cite`，参考文献使用 `IEEEtran.bst`。
4. 多附录使用 `\appendices`；取消原来强制另起一页的 `\clearpage`。
5. 更新图片路径；将正文中硬编码的 Section 3 改成自动交叉引用。
6. 两张表改用模板允许的跨栏 `table*`，数值与逐项 `(assumed data)` 标记分行显示；不缩小表格字号。
7. 为较长的候选预算列表增加允许换行点，不更改数值；用 IEEEtran 提供的 `\IEEEtriggeratref` 平衡末页参考文献的两栏。

不引入 `geometry`、`titlesec`、`caption`、CoRL 样式或自定义页边距、全局字号、行距、栏宽及标题样式。

## 编译

在 `CCDC` 根目录运行：

```powershell
.\scripts\build-1020.ps1
```

默认工具链：`D:/app/latex/texlive/2026/bin/windows/` 下的 `pdflatex.exe` 和 `bibtex.exe`。执行 LaTeX、BibTeX、LaTeX、LaTeX；所有生成文件位于 `../../output/1020/`。

## 当前状态

- 迁移后全文 8 页，包含正文、图表、附录和参考文献；符合已整理的初稿总页数上限。
- 样式文件和三张图须与来源文件保持 SHA256 一致。
- 编译后的页面预览位于 `../../output/1020/review/`。
- 此版本是格式迁移稿。原稿中的假设数据、TBD 与作者占位信息全部保留，尚需后续内容定稿；图中文字也沿用来源图片，后续可另行优化其双栏可读性。

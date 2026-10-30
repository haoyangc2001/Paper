# CCDC 1029：无附录投稿版本

本版本于 2026-10-29 从 `1020` 完整复制建立。正式编译入口为 `main.tex`，输出为 `../../output/1029/main.pdf`。当前 PDF 共 **7 页（包括参考文献）**，符合官网最多 8 页的篇幅要求。

同日完成表达润色：使用 `latex-paper-en` 的 expression 模块检查并作 39 处局部措辞调整。论证、定义、所有数值、数学表达、引用和图表环境未变；编译后仍为 7 页。逐项修改记录为 `submission/1029/expression_polish_review.tex`，新版页面预览为 `output/1029/expression_polish/`。下方“本次改动”及原版本核查记录描述复制与去附录阶段。

## 本次改动

- 仅从编译正文中移除参考文献后的附录；正文、数据、参考文献、图表内容及官方模板格式保持一致。
- 8 个图片路径由 `images/1020` 切换为 `images/1029`；7 个图环境与 3 个表环境仍为单栏。
- 不包含附录的正文未发现悬空的附录引用。
- `1020` 的原始文件保留；完整源稿另存于 `archive/main_1020_with_appendices.tex`，旧版说明另存于 `archive/README_1020_history.md`。

## 资料位置

沿用 CCDC 的平行目录布局，以下目录均建立了本版本副本：

| 内容 | 相对 CCDC 根目录的位置 |
| --- | --- |
| LaTeX 主稿、样式、BibTeX 文献库 | `latex/1029/` |
| 正文引用图片及相关素材 | `images/1029/` |
| 实验与绘图参考数据 | `data/1029/` |
| 文献材料 | `references/1029/` |
| 原有脚本快照 | `scripts/1029/` |
| 投稿要求及其他材料 | `submission/1029/` |
| 编译结果 | `output/1029/` |

复制前逐文件比对了 385 个文件的 SHA256，清单为 `submission/1029/copy_manifest.json`。清单记录复制时的原始状态；主稿、当前说明和编译产物随后按本版本需要更新，不应再要求这些目标文件与旧版相同。

`scripts/1029/` 中的脚本是原始工具快照，可能仍含旧版路径，使用前需检查。当前可直接使用的编译入口是 `scripts/build-1029.ps1`。归档的要求、审查记录及旧版预览只用于追溯，不能视为本次检查结论。本次版面检查图位于 `output/1029/review_no_appendix/`。源 `plan/` 为空，未复制计划文件。

## 编译

公式排版更新：所有独立公式采用居中的 equation 环境。公式（2）的三个约束分为三行、共用编号（2）；原无编号的残差 W 补为公式（7）。七个公式编号连续，并添加独立标签；公式内容与正文未改变，编译后仍为 7 页。最新页面预览位于 `output/1029/equation_layout/`。

2026-10-29 排版更新：英文润色已保存为本地提交 `e543a9f`。随后放宽实验小节之间的浮动体屏障，保留主章节边界与单栏图表，并平衡参考文献末页双栏。正文、公式、数据与图表内容未改；结论完整位于第 6 页，参考文献在第 7 页。最新预览位于 `output/1029/layout_optimization/`，不再以先前的表达润色预览代表当前版面。

在 CCDC 根目录运行：

```powershell
.\scripts\build-1029.ps1
```

编译入口调用本机安装的 `latex-paper-en/scripts/compile.py`，由 latexmk 调度 PDFLaTeX 与 BibTeX。默认 Python 为 `D:/codesoftware/Miniconda/python.exe`，TeX Live 为 `D:/app/latex/texlive/2026/bin/windows`。迁移到其他设备时，可通过脚本参数调整路径。

## 篇幅核查

2026-10-30 作者栏更新：按用户要求移除 Jingduo Pan，采用官方模板的三个独立作者信息块，依次为 Haoyang Cao、Yuhua Guo、Bai Xue。根据用户提供的参考截图，Yuhua Guo 同时署名 Space Star Technology Co., Ltd, China Academy of Space Technology 及 State Key Laboratory of Space Information System and Integrated Application, Beijing Institute of Satellite Information Engineering；两个完整单位名称保留，共用的 Beijing, 100095, China 仅在末尾列出一次。其余作者地点均为 Beijing, China，作者栏无占位符；通讯作者仍待确认。来源及核查记录见 `submission/1029/作者信息来源.md`。编译仍为 7 页，日志无溢出或未定义引用警告；最新全页预览位于 `output/1029/author_information/`，此前排版预览不代表当前版面。

2026-10-29 重新访问 CCDC 官方投稿说明：http://www.ccdc.neu.edu.cn/cgtj/list.htm 。初稿最多 8 页；终稿超过 6 页需付超页费 200 元/页，最多 8 页。当前按整份 PDF 计页，不假定参考文献豁免。7 页版本处于上限内；若最终按 7 页提交，按现有说明超页费为 200 元。

本次仅处理版本复制与附录移除，不代表数据、作者信息或投稿材料已经完成最终核验。

2026-10-30 后续更新：第二作者的共用地点简化为 `Beijing, China`，移除邮编，两个完整单位名称保留。在引言贡献列表之后加入实现代码的公开地址 `https://github.com/haoyangc2001/NaVILA_CaMRA`，按用户要求采用正式公开的表述。

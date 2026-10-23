# CCDC 2027 投稿工作区

按 CORL 的主要目录布局建立，当前已下载并验证 CCDC 官方 LaTeX 模板。

## 目录

```text
CCDC/
├── data/           实验数据
├── images/         论文图片
├── latex/
│   ├── 1020/       2026-10-20 迁入的论文正文与官方样式
│   └── template/
│       └── Conference-LaTeX-template_10-17-19/  官方原始模板
├── output/
│   ├── 1020/       当前论文编译输出
│   └── template/   官方示例编译输出
├── plan/           修改计划与工作记录
├── references/     参考文献材料
├── scripts/        编译及辅助脚本
└── submission/     官方下载包及投稿材料
```

论文版本放入 `latex/<版本>/`，编译结果放入 `output/<版本>/`，与 CORL 的版本组织方式一致。`latex/template/` 是保留不变的官方示例。当前论文版本为 [1020](latex/1020/README.md)，内容迁自 `CORL/latex/0914/main.tex`，相关图片位于 `images/1020/`。

## 模板来源

- 下载日期：2026-10-20。
- CCDC 官方投稿说明：http://www.ccdc.neu.edu.cn/firstSubmit/list.htm
- 官方 LaTeX 压缩包：http://www.ccdc.neu.edu.cn/_upload/tpl/01/4f/335/template335/htmlRes/download/conference-latex-template.zip
- 本地原始包：`submission/conference-latex-template.zip`。
- SHA256：`4C0539EED7BD4221D026382A56C57EC1668C55A8CA72B581F7A28654401DEDC9`。
- 主文件：`latex/template/Conference-LaTeX-template_10-17-19/conference_101719.tex`。
- 样式：模板自带 `IEEEtran.cls`，采用 IEEE 双栏会议格式。

压缩包中的目录名含 `10-17-19`，这是当前 CCDC 官方提供的通用 IEEE 模板的原始名称，并非下载了另一届会议模板。原始模板文件保持不变。

## 编译

在本目录运行：

```powershell
.\scripts\build-template.ps1
```

脚本使用本机 `D:/app/latex/texlive/2026/bin/windows/pdflatex.exe`，连续编译两次，结果为 `output/template/conference_101719.pdf`。

编译当前论文：

```powershell
.\scripts\build-1020.ps1
```

结果为 `output/1020/main.pdf`。2026-10-23完成数据一致性核查，换入新版多方法点区间图，同步消融重复基准和预算曲线；全文仍为8页（含附录和参考文献）。总览图在第4页，新结果图和主表在第5页，消融表和预算图在第6页。模板格式保持不变，假设数据性质、TBD和作者占位仍保留。修改记录见[数据一致性核查](submission/1023数据一致性核查.md)。

## 本届篇幅说明

2026-10-20 核实的官方中文投稿页规定：初投最多 8 页；终稿常规篇幅 6 页，超过 6 页每页加收 200 元，终稿最多 8 页。建议按全文 6 页准备，参考文献在未获明确豁免前计入总页数。应使用官方格式重排后统计页数，不能直接按 CoRL 单栏稿页数比较。投稿前以官网及系统最新说明为准。

详细规则、官方来源、待确认事项及本稿篇幅建议见 [CCDC 2027 投稿要求与篇幅规划](submission/CCDC2027投稿要求与篇幅规划.md)。官方中英文征稿 PDF 已归档到 `submission/`。

## 实验与绘图审查

2026-10-20 对当前 1020 稿进行了证据与图表审查，见 [结果缺口与绘图建议](submission/1020审稿视角_结果缺口与绘图建议.md)。六篇相关论文的本地原文、图页预览、重点原图及来源说明见 [绘图参考阅读集](references/figure_review_20261020/README.md)。本轮联网审批服务故障，归档文件为已有本地文献的原样副本，未冒称新下载。

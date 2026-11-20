# ICCRT 2027 稿件工作区

## 当前版本：2026-11-19 EECR内容 / ICCRT版式六页适配版

按作者最新指示，采用EECR已精简的六页稿内容，在ICCRT目录建立独立版本，按ICCRT官网Word模板主要版式适配LaTeX。最终PDF为 **6页，含16篇参考文献**，无超页费对应的额外页数。CCDC、EECR及下方旧七页副本均保留。

- 当前投稿PDF：[output/1119_6page/main.pdf](output/1119_6page/main.pdf)
- LaTeX主稿：[latex/1119_6page/main.tex](latex/1119_6page/main.tex)
- 文献库：[latex/1119_6page/references.bib](latex/1119_6page/references.bib)
- 独立图片：`images/1119_6page/`，8份文件与EECR源图哈希完全相同。
- 模板规格与来源：[template_adaptation.json](output/1119_6page/qa/template_adaptation.json)
- 内容、哈希与页数检查：[verification.json](output/1119_6page/qa/verification.json)
- 逐页检查：[visual_review.md](output/1119_6page/qa/visual_review.md)
- 构建：`& ./manuscripts/ICCRT/scripts/build-1119_6page.ps1`

正文与EECR六页稿相比未删改任何句子、公式、数据、图表、图注或引文。仅调整格式，合并5处相邻同主题段落，并独立复制图片与文献库。源文档逐字符核对时只忽略空白、图片路径和末页平衡命令。

官网只提供Word模板，本地保留其原文件；本稿为依据该模板主要规格制作的LaTeX适配，**不是官方LaTeX模板**。纸张612×792 bp，正文页边距上54、下72、左右45.35 bp，两栏间距18 bp，正文10 bp、摘要9 bp、图注及参考文献8 bp，标题24 bp。科学图片字形和现有数学字体系统保留，不能称为Word输出逐像素一致。检查记录说明该边界。

**状态：未提交。** 先前浏览器表单仍需选择新版PDF，不能继续使用旧的`output/1119/main.pdf`。尚未取得ICCRT投稿编号，未将其登记为已投稿。浏览器工具的附件目录访问限制需要作者手动选择该新版附件。

当前PDF SHA256：`774f38844677f0a75d0f29c806d5e9457bed0dd811b0707d0e8d6afa24e20022`。

## 历史：2026-11-19 CCDC七页原稿副本（已由上方六页稿替代）


2026-11-19 按作者授权，以 CCDC 1029 原稿建立独立副本。原 CCDC 文件保留。当前为待上传状态，尚未取得投稿编号。

| 位置 | 内容 |
| --- | --- |
| `latex/1119/main.tex` | 与 CCDC 1029 主稿逐字节一致 |
| `latex/1119/references.bib` | 与 CCDC 1029 文献库逐字节一致 |
| `latex/1119/IEEEtran.cls` | 原稿使用的 IEEEtran 类文件 |
| `images/1029/` | 原稿实际引用的八份图片，保留相对路径 |
| `output/1119/main.pdf` | 与 CCDC 1029 PDF 逐字节一致的七页版本，含23篇参考文献 |
| `template/conference-template-letter.docx` | 2026-11-19 从本届官网下载的原始全文模板 |

本次没有改动正文、数值、图表、署名、引文或结论，也没有重新编译。PDF SHA256：`aab3bebc77171fcf6606fe451cf3897b55965a8a6fd932f4204d47d2e11cbcc2`。

本届官网只提供 Word 模板，没有公开 LaTeX 下载链接。下载模板正文为 IEEE 风格、US Letter、双栏、数字编号引用；CCDC 原稿也使用 IEEEtran conference、Letter 双栏。以上是版式对照，不代表两个模板文件相同。初投系统只要求上传 PDF，未要求源文件。

官网要求全文不少于4页，页数包括标题、作者、单位、图片及参考文献。基础注册费包含6页，超页420元/页。保留当前7页预计产生420元超页费，注册类别和最终金额以录用后会务要求为准。本次未付款。

## 已核对的会议要求

- 第五届控制与机器人技术国际会议，2027-06-11至13，中国乌鲁木齐。
- 投稿截止2027-01-20，计划通知2027-02-20，注册截止2027-03-15。
- 研究方向选择 Track 2: Robotics and Autonomous Systems。
- 会场为瑞豪国际酒店，乌鲁木齐市新市区河南西路89号。
- 秘书 Ms. Jane Li，邮箱 `iccrtconference@yeah.net`。
- 公开页面要求英文、原创、未发表，未见明确禁止同时在审的条款；不能据此确认允许并行投稿。本次可见初投表单只有作者系统联系人勾选项，没有未投他刊或他会的声明。
- 本届官网未明确论文集出版方，不能用2026历史页的IEEE Xplore说明替代2027出版承诺。官网称盲审，当前初投表单没有要求匿名稿。

官网：https://www.iccrt.org/ 。投稿要求：https://www.iccrt.org/sub.html 。费用：https://www.iccrt.org/reg.html 。模板：https://www.iccrt.org/download/conference-template-letter.docx 。会议系统：https://easychair.org/conferences/?conf=iccrt2027 。

## 当前操作状态

已登录并填入三名作者、作者顺序、单位、标题、原稿摘要及三个关键词。Haoyang Cao 为唯一系统联系人，沿用作者此前确认的邮箱与联系人设置。第二作者系统单位使用此前EECR已确认的120字符缩写版本，PDF中的完整单位不变。

自动选择附件被浏览器工具的目录访问规则拒绝，需要作者在当前页面手动选择 `output/1119/main.pdf`。尚未点击最终提交，不能将本稿写为已投稿。选好附件后应核对浏览器中文件大小与哈希，再提交并核对服务器回执，最后追加到 `doc/投稿记录.xlsx`。

# EECR 2027 独立准备目录

## 2026-11-18：正文六页精简版（当前入口）

在16篇参考文献的1118版本上，按用户明确授权精简正文并逐段核对含义与逻辑。最终PDF恰好6页（含参考文献）。公式、数值、图表和图注保持，官方字号与版式保持；无缺失引用、Overfull或页外文字。旧版本保留。

- [六页PDF](output/1118_concise/main.pdf) / [LaTeX主稿](latex/1118_concise/main.tex)
- [逐段语义与逻辑复核](output/1118_concise/qa/semantic_review.md) / [交付说明与修改对照入口](../../doc/research/20261118_eecr_prose_concision/handoff.md)
- 构建：`& ./manuscripts/EECR/scripts/build-1118_concise.ps1`
- 验证：`& D:/codesoftware/Miniconda/python.exe ./manuscripts/EECR/scripts/verify-1118_concise.py`

## 2026-11-18：参考文献精简版（历史）

按用户本次明确授权建立1118独立修订：引用从23篇减至16篇，删除两处对应背景例句；方法、实验、公式和图表保持原样。编译后仍为7页，尚未达到6页免超页费目标。1114及下列迁移记录继续作为历史版本保留。

- [修订LaTeX](latex/1118/main.tex) / [修订PDF](output/1118/main.pdf)
- [往届数量调研与7项删除理由](../../doc/research/20261118_eecr_reference_reduction/findings.md)
- 构建：`& ./manuscripts/EECR/scripts/build-1118.ps1`
- 检查：`& D:/codesoftware/Miniconda/python.exe ./manuscripts/EECR/scripts/verify-1118.py`

## 2026-11-14：原样迁移记录（历史）

2026-11-14：按用户“按照之前的要求同样整理一下EECR”的授权，参照ICACAR目录方式，从CCDC当前1029版本建立EECR的1114独立副本。此次授权仅适用于新建EECR准备材料，原CCDC及ICACAR文件保持只读。

本地迁移、编译和版面核对已完成。正文、公式、数据、表格、8份图像及23篇参考文献保持原样；PDF为7页，符合EECR公开的4–10页范围。注册费包含6页，第7页另收费450元或60美元。

| 入口 | 内容 |
| --- | --- |
| [LaTeX主稿](latex/1114/main.tex) | 署名稿；仅调整会议注释及图片目录 |
| [论文PDF](output/1114/main.pdf) | 7页，本地核对版本 |
| [官方模板](latex/template/download_manifest.json) | 从官网取得的LaTeX ZIP、Word模板、征稿PDF及哈希 |
| [模板比对](latex/template/template_comparison.json) | 官方IEEEtran.cls与CCDC、ICACAR完全相同 |
| [投稿要求与交接](submission/1114/requirements_and_handoff.md) | 日期、渠道、页数、费用、政策及待核实项 |
| [投稿字段草稿](submission/1114/submission_metadata.md) | 题目、作者顺序、单位、邮箱、摘要、关键词 |
| [编辑咨询邮件草稿](submission/1114/editor_inquiry_draft.txt) | 已披露CCDC在审状态的政策咨询，未发送 |
| [核查证据](output/1114/qa/verification.json) | 原样迁移、PDF逐页比对、字体和引用核查 |
| [目视记录](output/1114/qa/visual_review.md) | 对应当前PDF哈希的版面核对 |
| `submission/1114/official_sources/` | 官网原始页面、正文、费用图片、URL、时间及哈希 |

在仓库根目录运行：

```powershell
& ./manuscripts/EECR/scripts/build-1114.ps1
& D:/codesoftware/Miniconda/python.exe ./manuscripts/EECR/scripts/verify-1114.py
```

构建通过latex-paper-en的compile.py，默认关闭shell escape。准备脚本prepare-1114.py仅用于首次下载及迁移，已有材料时主动停止，避免覆盖原始证据。

EECR官网：<https://www.eecr.org/>；在线投稿入口：<https://easychair.org/conferences/?conf=eecr2027>。计划投稿截止2026年12月10日、全文通知2027年1月10日、注册截止2027年1月30日；会期2027年4月23–25日，深圳。

当前完成的是本地准备。公开页面未写明同时投稿许可，CCDC一侧的IEEE要求仍需满足；CCDC来源README还保留部分统计为内部推断示例的说明，必须由作者核实后决定正式投稿。此次未发送邮件、提交论文、注册、付款或撤回其他会议稿件。匿名与独立标题页要求未在已查页面明确，因此未机械套用ICACAR的匿名材料要求。

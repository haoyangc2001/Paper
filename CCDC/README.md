# CCDC 2027 论文工作区

当前版本为[1029无附录稿](latex/1029/README.md)：[LaTeX正文](latex/1029/main.tex)、[论文PDF](output/1029/main.pdf)。PDF共7页，含23篇参考文献；图表为单栏。原1020完整版及官方模板保留。

## 目录

| 目录 | 内容 |
| --- | --- |
| `latex/1029/` | 当前源稿、BibTeX及IEEE样式 |
| `output/1029/main.pdf` | 最新编译PDF |
| `images/1029/` | 当前正文图片 |
| `data/1029/` | 绘图数据、配置及数据来源 |
| `references/1029/` | 参考论文及来源 |
| `scripts/` | 编译、绘图和核查脚本 |
| `submission/1029/` | 投稿资料、作者来源、数据替换说明 |
| `latex/1020/` | 历史完整源稿 |
| `latex/template/` | 官方原始模板 |

## 编译

在仓库根目录运行：

```powershell
& CCDC/scripts/build-1029.ps1
```

结果写入`CCDC/output/1029/main.pdf`。脚本通过本机`latex-paper-en/scripts/compile.py`调用latexmk；默认使用`D:/codesoftware/Miniconda/python.exe`及`D:/app/latex/texlive/2026/bin/windows`，可通过脚本参数调整。编译中间文件可重新生成，不需要作为正文资料保存。

## 数据与版本

当前名称为CaMRA，历史数据中的HD-MCRA标识通过`scripts/method_names.py`映射。数据来源、推断台账及真实数据替换说明保留在data和submission目录。当前新增统计仍有内部推断示例，格式及语言定稿不等于完成真实实验验证。

`scripts/1029/`为复制时的工具快照，部分路径指向旧版本；当前编译使用上面的独立入口。

## 投稿资料

- [本届投稿核查](submission/1029/CCDC2027_投稿合规核查_20261030.md)
- [官方投稿要求及篇幅记录](submission/1029/CCDC2027投稿要求与篇幅规划.md)
- [新增参考论文](references/1029/literature_expansion_20261030/README.md)
- 官方页面：http://www.ccdc.neu.edu.cn/firstSubmit/list.htm

此前记录为初稿最多8页、终稿超过6页需付超页费；正式提交以本届系统最新要求为准。

## 过程资料

2026-10-31清理的计划、排版预览和阶段报告保存在本机[集中备份](../.local/cleanup_20261031/README.md)。原文件路径与SHA-256均在备份清单中；历史资料中的过程文件链接可据此恢复。

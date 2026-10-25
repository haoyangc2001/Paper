# 环境与来源基线

2026-10-25。使用本机既有 Conda：`D:/codesoftware/Miniconda/python.exe`。

- Python 3.12.4；Matplotlib 3.10.8；NumPy 2.2.4；PyMuPDF 1.27.2.3；Pillow 12.0.0。
- Poppler：`C:/Users/smile/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/poppler/Library/bin/pdftoppm.exe`。
- 未安装或升级依赖，无远程服务器调用。
- 当前论文：`CCDC/latex/1020/main.tex`，消融表 `tab:ablation_draft`，25个显示值与生成脚本的 `ABLATION_ROWS` 相符。
- 检查发现现有工作簿没有消融工作表；它只作主结果锚点的来源。消融单裕度和去层级行以匹配的论文/源码为依据，不误称Excel也保存了这些值。来源工具已按这一事实修正，未改动原文件。
- 成本物理单位/绝对预算、逐试验记录和seed分配仍未落实；本图采用已确认的假设汇总协议。

详细来源哈希和表格提取结果见 `baseline.json`，后续用它检查原文件未被改写。

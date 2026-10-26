# 完整内部模拟稿：2026-10-26

本轮直接更新原稿 `CCDC/latex/1020/main.tex` 和 `CCDC/output/1020/main.pdf`。正文按完整论文的句式撰写；新增配置、计数、SD及区间仍是内部示例，不是本轮测量结果。没有进行训练、采样或真机实验。旧人工区间及历史数据保留。

## 入口与文件

| 文件 | 用途 |
|---|---|
| `../../../plan/manuscript_completion_20261026_plan.json` | 26项顺序执行记录；包含失败及修复历史 |
| `gap_register.json` | 28项缺口，27项写作闭合，作者身份保留未知 |
| `method_contract.md` / `statistical_contract.md` | 理论/工程与统计语义边界 |
| `../../data/manuscript_completion_20261026/protocol.json` | 成本、几何、低高层配置、场景、样本、校准、扰动和硬件设定 |
| `../../data/manuscript_completion_20261026/inference_ledger.jsonl` | 67项配置/结果决策及依据 |
| `../../data/manuscript_completion_20261026/main_result_ledger.jsonl` | 105个主比较指标的来源、中心/端点、关联及替换记录 |
| `../../data/manuscript_completion_20261026/continuous_sd.csv` | 42个逐项确定的连续指标SD及理由；用于计算CI |
| `../../data/manuscript_completion_20261026/margin_sweep.csv` | 16设置×5指标逐项理由；四锚点与主表/消融相同 |
| `stage_reviews.md` | 六次阶段审查及修正 |
| `full_audit.json` / `figure_manifest.json` | 数值一致性、绘图输入与论文图片哈希 |
| `layout_review.md` | 10页全部视觉检查，投稿页数仍超限 |
| `replacement_map.md` | 真实数据到位后逐项替换路径 |

## 复算和编译

在项目根目录运行：

```powershell
& D:/codesoftware/Miniconda/python.exe -B CCDC/scripts/audit_manuscript_completion_20261026.py contract
& D:/codesoftware/Miniconda/python.exe -B CCDC/scripts/audit_manuscript_completion_20261026.py main
& D:/codesoftware/Miniconda/python.exe -B CCDC/scripts/audit_manuscript_completion_20261026.py calibration
& D:/codesoftware/Miniconda/python.exe -B CCDC/scripts/audit_manuscript_completion_20261026.py ablation_budget
& D:/codesoftware/Miniconda/python.exe -B CCDC/scripts/audit_manuscript_completion_20261026.py margin
& D:/codesoftware/Miniconda/python.exe -B CCDC/scripts/audit_manuscript_completion_20261026.py hardware
& D:/codesoftware/Miniconda/python.exe -B CCDC/scripts/audit_manuscript_completion_20261026.py results
& D:/codesoftware/Miniconda/python.exe -B CCDC/scripts/audit_manuscript_completion_20261026.py full
& D:/codesoftware/Miniconda/python.exe -B CCDC/scripts/audit_manuscript_completion_20261026.py figures
& D:/codesoftware/Miniconda/python.exe -B CCDC/scripts/audit_manuscript_completion_20261026.py wording
& CCDC/scripts/build-1020.ps1
```

上述`main/calibration/...`复算的是本轮内部示例，尚不是实测数据导入程序。不得将真实数据替换进去后再用旧源生成器覆盖它。图二脚本保留单独的empirical入口和来源校验；切换时需填真实样本协议，不只改标签。

## 图像新版本

- 图二输入：`绘图/20261023_多方法指标对比/data/manuscript_completion_20261026/`；输出为同项目`output/manuscript_completion_20261026/metrics_comparison.pdf`，220×90mm。
- 预算图输入：`绘图/20261025_预算扫描图/data/manuscript_completion_20261026/`；输出为同项目`output/manuscript_completion_20261026/budget_sweep.pdf`，180×50mm。
- 两图新版本已复制到论文既有图片入口。其余五图（硬件图含两张照片）没有重绘。

当前共10页；作者/单位/邮箱仍是原匿名模板。真实数据、真实配置和作者身份替换，以及压至投稿上限，均不能由本轮内部一致性检查代替。未自动Git提交或推送。

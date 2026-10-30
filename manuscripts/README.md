# 论文写作资料

本目录集中保存论文源稿、最终 PDF、参考文献、投稿资料、绘图脚本及其数据。既有 `RC-PPO/` 代码项目保留在仓库根目录。

| 目录 | 内容 |
| --- | --- |
| [CCDC](CCDC/README.md) | 当前 1029 投稿稿件、历史 1020 完整稿、IEEE 模板及配套资料 |
| `CORL/` | 早期稿件、原审稿意见、参考论文和历史结果脚本 |
| [figures](figures/README.md) | 各图的独立项目；保留原有子目录和 `data/` |

当前入口：[LaTeX 主稿](CCDC/latex/1029/main.tex)、[最终 PDF](CCDC/output/1029/main.pdf)。

在仓库根目录运行：

```powershell
& ./manuscripts/CCDC/scripts/build-1029.ps1
```

历史绘图文档中的命令若以 `CCDC/`、`CORL/` 或 `figures/` 开头，应从本目录运行；`figures/` 已重命名为 `figures/`。脚本内部按文件自身位置定位写作资料，不依赖启动位置。

2026-10-31目录迁移如下：

| 原仓库相对路径 | 当前仓库相对路径 |
| --- | --- |
| `CCDC/` | `manuscripts/CCDC/` |
| `CORL/` | `manuscripts/CORL/` |
| `figures/` | `manuscripts/figures/` |

历史数据来源清单、哈希记录、阶段审查和备份保留生成时的原路径，查询时按上表映射；它们不是当前运行入口。原计划及已清理过程文件的恢复入口位于仓库根目录的 [.local/cleanup_20261031](../.local/cleanup_20261031/README.md)，选择恢复时也需应用上述映射。

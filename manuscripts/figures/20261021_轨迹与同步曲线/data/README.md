# 轨迹与同步曲线：最终绘图数据

核对日期：2026-10-25。项目原有 `data/` 已包含最终数据，无需再复制另一套同名文件。已逐文件核对：根目录中的12个原始、派生和说明文件与 `versions/v2/` 一致，其中6个成图派生/配置文件也与 `../output/publication_v4_final/data/` 一致。

- `raw_hd_mcra.csv`、`raw_rcppo.csv`：逐时刻原始假设记录。
- `derived_hd_mcra.csv`、`derived_rcppo.csv`：轨迹、成本、间隙和跟踪误差等派生绘图量。
- `scenario.json`、`metadata.json`：场景、障碍、起终点、预算、两个裕度、成本定义和数据性质。
- `events.csv`：各方法自己的A/B等事件时刻。
- `summary.json`：各方法成本、最小间隙、到达时间等汇总。
- `protocol.md`、`schema.md`、`assumptions.md`、`revision_notes.md`：字段、假设与版本说明。

当前最终图：`../output/publication_v4_final/trajectory_sync.pdf`。`versions/`、`pilot/`、`pilot_attempt1/`、`attempts/` 是历史/试画资料，不要误作当前最终输入。

现有 `scripts/render_publication.py` 默认读取 `data/versions/v2/`，支持 `--data-dir`。若修改本目录根部的数据，可在项目根目录运行：

```powershell
& 'D:/codesoftware/Miniconda/python.exe' scripts/render_publication.py --data-dir data --output-dir output/updated_preview
```

完整说明见 `../图表说明与使用手册.md`；不要覆盖已验收最终输出。本次仅补充数据入口说明，没有变更任何轨迹或指标。所有记录仍为假设数据，不是实测或物理仿真输出。

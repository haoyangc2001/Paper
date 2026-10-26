## 2026-10-26 成本口径补充
当前论文采用本项目既有3.6/3/20/2代理成本公式作为统一a.u.定义。图三预算仍85、数据和图像未改，它是独立短诊断地图，不混入主比较120/140/190预算下的三场景均值。旧文“不可比较成本尺度”属于定义尚未统一时的历史状态。
原数据仍为合成轨迹；同尺度数学定义不等于已恢复真实机器人日志。替换指南：CCDC/submission/manuscript_completion_20261026/replacement_map.md。

# 轨迹与同步曲线：论文图版式与数据
已完成逐点数据构造、计算校验、四次阶段审查、原始数据与排版修订、独立复现。最新数据为v2；当前推荐版式为publication_v4_final，旧版v3保留。

所有运动记录均为逐点编写的假设示例，不是实测、物理仿真或策略输出。只用于审阅图形和论证逻辑。

详细阅读：[图表说明与使用手册](C:/Users/smile/Desktop/paper/绘图/20261021_轨迹与同步曲线/图表说明与使用手册.md)。包含逐部分读图、原始字段与计算公式、重绘命令、修改对照表、真实日志适配、论文使用建议及常见问题。

## 文件入口

- [当前论文版式PNG](C:/Users/smile/Desktop/paper/绘图/20261021_轨迹与同步曲线/output/publication_v4_final/trajectory_sync.png)
- [当前论文版式矢量PDF](C:/Users/smile/Desktop/paper/绘图/20261021_轨迹与同步曲线/output/publication_v4_final/trajectory_sync.pdf)
- [当前配套中英图注](C:/Users/smile/Desktop/paper/绘图/20261021_轨迹与同步曲线/output/publication_v4_final/caption.md)
- [本次版式整理与检查](C:/Users/smile/Desktop/paper/绘图/20261021_轨迹与同步曲线/output/publication_v4_final/revision_review.md)
- [旧版带说明PNG](C:/Users/smile/Desktop/paper/绘图/20261021_轨迹与同步曲线/output/final/trajectory_sync_assumed.png)
- [旧版带说明矢量PDF](C:/Users/smile/Desktop/paper/绘图/20261021_轨迹与同步曲线/output/final/trajectory_sync_assumed.pdf)
- [RC-PPO原始数据](C:/Users/smile/Desktop/paper/绘图/20261021_轨迹与同步曲线/data/versions/v2/raw_rcppo.csv)
- [HD-MCRA原始数据](C:/Users/smile/Desktop/paper/绘图/20261021_轨迹与同步曲线/data/versions/v2/raw_hd_mcra.csv)
- [旧版中英图注](C:/Users/smile/Desktop/paper/绘图/20261021_轨迹与同步曲线/output/caption.md)
- [当前数据审查](C:/Users/smile/Desktop/paper/绘图/20261021_轨迹与同步曲线/output/data_audit_v2.md)
- [视觉审查](C:/Users/smile/Desktop/paper/绘图/20261021_轨迹与同步曲线/output/visual_audit.md)
- [计划与工作记录](C:/Users/smile/Desktop/paper/plan/trajectory_sync_figure_20261021_plan.json)

output/publication_v4_final/panels含当前四个独立面板的PDF与PNG；render_report.json记录本次输入与导出文件哈希，pdf_qa.json记录PDF检查。图内的假设数据标记、底部解释、共同接口说明及α/δ参数已移至配套图注；当前数据性质仍保留在图注、README、输入配置、生成记录与PDF元数据中。

output/final/panels及output/final_manifest.json是上一轮v3交付与历史清单，不代表修改后脚本的当前哈希；早期图、失败样本和旧数据快照保留用于追溯。

## 如何理解这张图
左图展示同场景路线，A/B事件与右侧三个时间面板对应。蓝色HD-MCRA保留较大空间余量，橙色RC-PPO更接近障碍；右侧依次展示实际机身间隙、累计成本和预算、线速度跟踪误差。

| 当前假设结果 | RC-PPO | HD-MCRA |
| --- | ---: | ---: |
| 首次到达时间/s | 13.2 | 14.8 |
| 路径长度/m | 4.4774 | 4.6122 |
| 采样最小间隙/m | 0.02758 | 0.14656 |
| 实际累计成本/示例单位 | 56.9085 | 61.5903 |
| 线速度RMSE/(m/s) | 0.02674 | 0.02803 |

两方法均未超过共同预算85。本例说明空间余量和预算计费关系，保留更安全但更慢更贵的取舍，不能证明总体性能提升或alpha减少预算违规。已有论文主表的成本单位仍未核实，不能把本例代理成本直接等同其88.3/91.7；0.19→0.08是另一种接口对照，不适用于本图。

A：各自首次通过x=1.9m；B：首次通过x=3.3m。时刻分别约6.0/6.8s、10.0/11.2s（RC-PPO/HD-MCRA），以冻结events.csv为准。不同方法不强制同时到达，误差末段只绘到最后一个有动作记录的区间起点。

## 数据与计算
原始文件七列：时间、指令vx/vy/yawrate、实际vx/vy/yawrate。全部原始数值显式写入静态CSV，可参考整个运动阶段及多时刻历史；未使用随机噪声、插值或规则脚本批量生成原始运动。实际速度为权威记录，位置由0.2s显式积分计算；yaw=0但全向平移允许路径转弯。

机身用半径0.20m圆形足迹；间隙对实体障碍计算，delta=0.12m为额外规划余量。名义取同命令理想速度，内部累计计费为1.18乘本方法名义累计成本。内部余额=85-内部计费，实际余额=85-实际成本。示例成本率：
`3.6 + 3*|v|² + 20*|v-v_cmd|² + 2*max(0,0.45-clearance)`，每步乘0.2s。

[协议](C:/Users/smile/Desktop/paper/绘图/20261021_轨迹与同步曲线/data/versions/v2/protocol.md)、[字段契约](C:/Users/smile/Desktop/paper/绘图/20261021_轨迹与同步曲线/data/versions/v2/schema.md)、[修订记录](C:/Users/smile/Desktop/paper/绘图/20261021_轨迹与同步曲线/data/versions/v2/revision_notes.md)给出详细单位、边界和终止规则。

## 环境与复现
主流程使用已安装的Conda环境 `D:/codesoftware/Miniconda/envs/py39`：Python3.9.21、NumPy1.26.0、Pandas2.2.3、SciPy1.13.1、Matplotlib3.9.4、Pillow11.2.1。无需安装新依赖。PDF渲染使用现有Poppler；只读PDF结构检查使用已有base环境PyMuPDF，不参与数据计算或绘图。

在PowerShell运行下面命令，输出目录须不存在，以保留历史产物：
```powershell
Set-Location -LiteralPath 'C:/Users/smile/Desktop/paper/绘图/20261021_轨迹与同步曲线'
& 'D:/codesoftware/Miniconda/Scripts/conda.exe' run -n py39 --no-capture-output python -B scripts/render_publication.py --data-dir data/versions/v2 --output-dir output/publication_rebuild_01
```

该命令校验输入、重算派生量、生成组合图及四张独立子图、更新配套图注并记录输入/输出哈希。替换数据时，将--data-dir指向新的数据目录。目录须包含scenario.json、metadata.json、raw_rcppo.csv和raw_hd_mcra.csv，字段契约同下文。脚本不生成原始数据。

预算文字与预算线、δ扩张、起终点、事件时刻、坐标范围、成本单位、最小间隙和耗时均读取配置或派生数据。α参与派生计费计算，并同步到图注；成本单位可通过scenario.json中的cost.axis_unit指定简写，否则使用cost.unit。真实场景若改变动力学、成本定义或采样协议，须同步适配计算部分；不能仅更改图中标签。

旧版仍可通过scripts/reproduce.py --output-dir output/rebuild_02复现。此次已验证output/legacy_check_after_v4：7项关键计算测试通过，派生CSV/JSON逐字节相同、PNG逐像素相同、PDF逐字节相同，原始输入与论文来源哈希保持不变。

分步执行计算/校验时，可以向新的输出目录导出，避免修改冻结快照：
```powershell
& 'D:/codesoftware/Miniconda/Scripts/conda.exe' run -n py39 --no-capture-output python -B scripts/compute_metrics.py --data-dir data/versions/v2 --output-dir output/metrics_check
& 'D:/codesoftware/Miniconda/Scripts/conda.exe' run -n py39 --no-capture-output python -B scripts/validate_data.py --data-dir data/versions/v2 --derived-dir output/metrics_check --report output/metrics_check/validation.json
```

## 审查与来源
四阶段报告在output/reviews：方案、小段、完整数据、成图。问题台账包含运动学术语、过早转弯、终止后冗余行、成本主张边界、排版重叠、误差幅值重复六项，均记录来源与修正证据。阶段审查后曾回到原始数据修订，并重新验收所有受影响任务。

主视觉参考为SEA-Nav Fig.4，PDF第6页，arXiv:2603.09460v1；本地出处及来源哈希在reference/source_manifest.json。借鉴其空间轨迹与时间曲线关联、灰色地图、细轴和事件标注，适配为右三排和蓝橙方法编码；没有复制其实验数据或CBF机制。参考同宽对照见output/reference_comparison.png。

## 未来替换真实日志
1. 新建数据版本，保存真实日志来源、trial ID、时间同步、坐标系及完整成本/预算定义；不要覆盖本次假设快照。
2. 将实际原始日志映射到字段契约；若真实足迹、采样、动力学或成本与本例不同，先修改协议和计算并验证，不能直接套示例参数。
3. 保留名义/实际配对和各自终止，重新计算全部指标、事件和统计；不能从本案例反推总体SR/CR/BVR。
4. 重绘并审查，只有相关数值全部有真实证据后，才将对应版本标识改成实测。论文与模板本次均未修改。



# 模块消融图：数据、绘制与维护说明

完成日期：2026-10-25。对应计划：`../../plan/ablation_figure_20261025_plan.json`。

## 当前版本：第二轮视觉优化

最新图片位于`output/refined_v2/`：[PDF](output/refined_v2/ablation_horizontal.pdf)、[PNG](output/refined_v2/ablation_horizontal.png)、[SVG](output/refined_v2/ablation_horizontal.svg)。旧`output/final/`保持原样。

本轮保持一行四图和全部原值，调整了以下内容：

- 左侧α/δ勾叉列替代w/o文本，勾表示启用，叉表示关闭。三行分别为×✓、✓×、××；双叉行仍对应RC-PPO。完整HD-MCRA两者均启用。
- 每栏顶部蓝色`Full`数值为同一完整方法参照，只标一次；12个彩色数值对应各消融点。
- 两处重点差值放在对应连线下方，单位pp表示百分点；点上数值是百分比，不是相对增幅。
- 空心点、标题和连线更轻，连接线采用对应行浅色。仍不表示误差区间，也没有改动接近点的位置。

尺寸88.9×43.5mm，最低字号6.5pt。状态、4个参考值、12个点值和2个差值均已核对；PDF文字无重叠裁切，150dpi PDF预览已检查。视觉优化阶段的数据与协议、旧图及论文哈希保持不变，10个跨图锚点一致。随后按用户要求已将本版接入CCDC当前稿件，见下方接入记录。

新样式单独位于`config/style_horizontal_v2.json`。原`config/style_horizontal.json`仍为脚本默认样式；要生成新版请明确传入`--style`，使用尚不存在的新输出目录：

```powershell
& 'D:/codesoftware/Miniconda/python.exe' -B '绘图/20261025_模块消融图/scripts/plot_ablation.py' --style '绘图/20261025_模块消融图/config/style_horizontal_v2.json' --output '绘图/20261025_模块消融图/output/refined_v2_rerun'

& 'D:/codesoftware/Miniconda/python.exe' -B '绘图/20261025_模块消融图/scripts/audit_render.py' '绘图/20261025_模块消融图/output/refined_v2_rerun'
```

新版增加`state_matrix`、`reference_labels`、`value_labels`、`legend`及按行配色配置。状态列必须与删除配置定义一致；只换数据时不要修改状态。图注由脚本生成，继续清楚标示全部为假设数据。

阶段复核见`reviews/refinement_v2.md`，跨图证据见`reviews/cross_source_refined_v2.json`。下文的`output/final/`是首版归档，数据入口、数据替换规则仍通用。

## 首版成果归档

最终图片位于 `output/final/`：

- `ablation_horizontal.pdf`：矢量图，后续接入LaTeX优先使用。
- `ablation_horizontal.svg`：保留文字和矢量元素，便于查看或必要的矢量编辑。
- `ablation_horizontal.png`：600dpi预览，约2100×980像素。
- `pdf_preview_150dpi.png`：从PDF渲染的阅读检查图。
- `caption.md`：中英文图注，关键差值由脚本从数据计算。
- `render_manifest.json`：输入哈希、脚本哈希、尺寸、文字框、每组对照的数值映射。
- `render_audit.json`：PDF尺寸、最低字号、文字边界和映射验收。

图为横向一行四面板，顺序是SR、Cost、BVR、CR；真实物理尺寸88.9×41.5mm，按IEEEtran单栏设计。字体6.5至7.2pt，PDF未出现出界或文字重叠。

当前数字全部是已有假设数据。25个中心值逐项审查后保留，没有生成新的试验、分布或误差棒；其中16个值进入主图，9个关联辅助值用于检查。独立重绘通过，最终图与重绘图的数值、PNG像素、PDF渲染和图注一致。

## 这张图表达什么

每个面板中的三行分别是去掉α、去掉δ、两者都去掉。最后一行对应使用同一冻结速度跟踪器的RC-PPO，不是去掉层级接口。

蓝色实心圆是完整HD-MCRA，同一面板三行重复使用同一个参照值；绿色菱形、紫色方形和橙色空心圆分别表示三个删除版本。细灰线只连接两组汇总中心，长度表示数值差距，不是误差线，也不代表随时间变化。

四图共12段连接、24个点标记，来自16个不同中心值。SR越大越好，BVR/CR越小越好。Cost需要和SR一起看，低成本可能伴随更早失败，不能单凭最小成本判优。

当前突出两处差值：去α时BVR增加13.4个百分点，去δ时CR增加11.3个百分点。四组假设值并非单调递进，不能据图声称每个模块单独加入都会全面改善。

CR最后一行的蓝点与橙环很近，因为15.0和15.6667相差很小，真实位置有局部重合；图中没有人为抖动坐标。不同面板采用各自单位和局部轴范围，不能横跨指标比较线长大小。

## 数据在哪里

| 文件 | 作用 |
| --- | --- |
| `data/reviewed_points.jsonl` | 唯一绘图数值输入，25条记录；`value`是采用值，`role=main`的16条进入图中 |
| `data/plot_protocol.json` | 面板和配置顺序、数据性质、锚点、统计口径 |
| `data/baseline_points.json` | 原值、论文显示值、来源和已有场景精度，保留作为历史基线 |
| `data/review_protocol.json` | 独立数据审查规则和锚点，不随绘图样式变化 |
| `data/reviewed_summary.json` | 审查发布时的12组差值和计数，便于阅读；绘图不会从这里读旧差值 |
| `data/source_manifest.json` | 15个既有来源的SHA256 |
| `reviews/point_review.jsonl` | 每个点独立的推理、关联来源、保留/修订结论，追加式维护 |

原始来源是当前论文消融表及`CORL/scripts/generate_experiment_results_v3.py`。现有Excel没有消融工作表，只用于主结果锚点检查；脚本不会依赖一个不存在的消融sheet。

完整方法/RC-PPO从三个场景等权平均保留全精度。论文显示一位小数，MC显示两位，Excel平均值储存到四位小数。例如成本91.666667−88.333333=3.333333，不应把显示91.7−88.3的3.4当精确差值。显示值和原值没有冲突。

## 改数据的方法

1. 在`data/reviewed_points.jsonl`找到对应`point_id`，如`NO_ALPHA_BVR`，修改`value`。来源、单位和假设性质也应准确记录。
2. 在`reviews/point_review.jsonl`追加该点下一版本记录，明确`revision`、`supersedes_revision`、采用值、原因及受影响点，不删原记录。
3. 回查同配置其他指标、其他配置、主表和已有图。完整方法/RC-PPO的10个锚点必须同步保持来源一致，不能只改本图。
4. 运行数据校验，再出图到新目录。差值标注和图注自动重新计算。
5. 更新审查摘要和README的具体结果解释，避免数据换了但文字仍沿用旧结论。

只改某个工作簿副本不会改变本图，因为唯一输入固定为本项目`data/`。目前协议按假设数据校验；未来实测替换需有试验/seed/地图记录，落实成本、预算和事件定义后再更新统计协议，不能只重命名数据性质。

如果未来增加误差区间，先明确区间代表SD、SE还是CI，并由相应试验数据计算。不要复用主结果图的场景假设区间，也不要把当前细连接线解释成误差棒。

## 改外观的方法

主要编辑`config/style_horizontal.json`：

- `width_mm`、`height_mm`：最终物理尺寸；保持单栏宽度时可调整高度。
- `font_pt`、`small_font_pt`、`title_font_pt`：最终字号。
- `colors`、`markers`：各配置颜色与点形；完整方法和RC-PPO与既有图保持一致。
- `axes`：面板位置、宽度、间距和纵向留白。
- `panels`：各指标标题、坐标范围和刻度。
- `annotations`：突出哪组差值，数值由数据自动计算。

绘图逻辑在`scripts/plot_ablation.py`。正常换数值不需要改绘图代码；改点的表现方式或增加新图层时才改代码。优先通过简短标签、共享行名和留白保证清楚，不通过虚移数据点避免重合。

## 运行与检查

在`C:/Users/smile/Desktop/paper`下使用PowerShell，运行既有Conda环境：

```powershell
& 'D:/codesoftware/Miniconda/python.exe' -B '绘图/20261025_模块消融图/scripts/check_ablation_data.py'

& 'D:/codesoftware/Miniconda/python.exe' -B '绘图/20261025_模块消融图/scripts/check_ablation_data.py' --cross-sources

& 'D:/codesoftware/Miniconda/python.exe' -B '绘图/20261025_模块消融图/scripts/plot_ablation.py' --output '绘图/20261025_模块消融图/output/revision_v2'

& 'D:/codesoftware/Miniconda/python.exe' -B '绘图/20261025_模块消融图/scripts/audit_render.py' '绘图/20261025_模块消融图/output/revision_v2'
```

输出目录必须不存在，避免覆盖历史版本。小批预览可加`--pilot`，它在完整四面板宽度中只显示SR和BVR，便于检验最终尺寸。

检查器验证25点身份、单位、概率边界、来源锚点、审查记录和渲染映射。`audit_render.py`进一步检查PDF尺寸、字号和文字框。仍需人工看PDF/PNG，尤其关注近邻点、文字留白和颜色区分。

`scripts/source_audit.py`只读取旧来源并导出来源基线；日常改图不需要重跑，以免覆盖最初来源指纹。`scripts/check_reproduction.py`比较本次final和reproduction_v1；新版本应相应指定或调整比较路径。

## 审查过程与后续使用

`reviews/stage_reviews.md`保留阶段复核；`global_data_review.md`解释所有数值的整体关系；`cross_source_check.json`及`cross_source_final.json`记录10个锚点和15个来源检查；`final_visual_review.md`记录视觉问题及修复；`reproduction_check.json`保存重绘证据。

保留了pilot_v1、pilot_review_v1、candidate_v1、final和reproduction_v1。第一版底部文字偏挤，后续将共同接口和预算说明移入图注，数值始终未变。最终版图注由脚本动态生成，防止换数据后仍带旧标注。

后续接入论文时使用最终PDF并配合图注，通常放在模块消融表附近。当前任务只完成独立绘图工程，尚未替换论文或编译整篇。论文和所有既有图的来源哈希保持不变。

## 后续论文接入记录（2026-10-25）

用户授权后，将`output/refined_v2/ablation_horizontal.pdf`原样复制到`../../CCDC/images/1020/ablation_horizontal.pdf`，在`../../CCDC/latex/1020/main.tex`的执行失配与模块分析小节以单栏图接入，标签`fig:ablation_margins`。图4位于编译稿第5页，原消融表保留，全文仍8页。

本次只调整论文插图、相关引用和图注，没有重新生成本项目图或修改数据。历史审查中的“论文哈希未变”对应插图前的视觉优化阶段，原基线继续保留。最新论文数据/图片来源核查由`../../CCDC/scripts/audit_results_1023.py check`执行，结果见`../../CCDC/output/1020/consistency_review/audit.json`。

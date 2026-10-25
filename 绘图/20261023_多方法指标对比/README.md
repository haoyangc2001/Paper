# 多方法、多场景指标对比图

## 最新尺寸修订：220×90mm

按用户反馈恢复旧四列图的90mm高度，宽度由旧图181mm扩到220mm，为第五列留空间。字体、点和线恢复接近旧版比例；105个点和210端点均不变。当前数据入口为[data/mc_extension_v2](data/mc_extension_v2/README.md)，当前成图为[PDF](output/mc_extension_v2/metrics_comparison_assumed.pdf)、[PNG](output/mc_extension_v2/metrics_comparison_assumed.png)、[SVG](output/mc_extension_v2/metrics_comparison_assumed.svg)。已同步论文，继续单栏等比插入；旧四列版与89×70mm五列版均保留。

维护当前版时将下方命令中的`mc_extension_v1`换为`mc_extension_v2`；MC核查脚本默认检查v2，指定`--data-dir`可检查历史v1。下方为第一版完成记录，其89×70mm尺寸和入口不再代表当前版。

日期：2026-10-23。2026-10-25 当前版本：`assumed-mc-extension-v1`。

## 当前五指标版

当前图片：[PDF](output/mc_extension_v1/metrics_comparison_assumed.pdf)、[PNG](output/mc_extension_v1/metrics_comparison_assumed.png)、[SVG](output/mc_extension_v1/metrics_comparison_assumed.svg)。沿用方法行、颜色及场景点形，顺序与表一一致：**SR、Cost、CR、BVR、MC**；图幅89×70mm，供论文单栏直接插入。已接入CCDC的1020稿件，为第5页图二；本轮前后全文均为9页。原四列成果保留在原`output/`位置，下面的原版说明作为历史记录。

新版共105点和210个端点：原84条记录不变，新增MC001—021；21个MC中心沿用原脚本，42个端点逐点人工推敲。MC是各次试验全程最小物理机身—障碍间距的均值，按全部试验统计，单位m，不是膨胀地图的距离，也不是δ。区间为假设估计范围，不是实测CI、SD、SE或单次试验分布。较大MC不保证较低碰撞率或更高成功率。

数据唯一入口为[data/mc_extension_v1](data/mc_extension_v1/README.md)中的JSONL；CSV和明细为导出副本。表一对应五项指标的三个场景等权平均，35项均已核对；不平均区间端点。21个点的关联依据在[推演明细](output/mc_extension_v1/推演明细.md)，阶段检查见[本轮审查](reviews/mc_extension_v1.md)。

从仓库根目录检查、重绘：

```powershell
& 'D:/codesoftware/Miniconda/python.exe' -X utf8 -B '绘图/20261023_多方法指标对比/scripts/audit_mc_extension.py' --self-test
& 'D:/codesoftware/Miniconda/python.exe' -X utf8 -B '绘图/20261023_多方法指标对比/scripts/plot_metrics_comparison.py' --data-dir '绘图/20261023_多方法指标对比/data/mc_extension_v1' --output-dir '绘图/20261023_多方法指标对比/output/mc_extension_v1'
```

改值先改新版JSONL，再重绘并将导出CSV同步到新版data目录；改样式改同目录protocol.json。每项指标配置单位、显示精度、范围、刻度和短标题；MC保留两位小数。真实数据替换应另建版本，填写真实来源、统计口径与区间估计方式，不能只改假设标签；旧中心一致性审查仅用于本次保留假设数据的版本。默认不传`--data-dir`仍读取原四列数据；维护新版时请明确指定上述路径，避免误绘旧版。

## 原四列版说明（历史）

本图是论文版式预览：七种方法、三个场景、四项指标，共84个点及84条横向区间。中心值沿用已有假设数据，168个左右端点逐点指定并保留理由。它们不是实测数据，也不是具有95%覆盖概率的置信区间。

当前排版为181 × 90 mm，相比初版111 mm高度缩短约19%。保留8 pt主体字号，压缩方法行之间的距离；PPO使用深浅紫色，CPPO使用深浅绿色，方法名称、点和误差线颜色一致。

## 查看结果

- [当前PNG](output/metrics_comparison_assumed.png)
- [当前矢量PDF](output/metrics_comparison_assumed.pdf)
- [每个点的数值、左右端点与理由](output/推演明细.md)
- [配套中英图注](output/caption.md)
- [分阶段审查记录](reviews/stage_reviews.md)

图的主体使用论文排版；假设性质保留在此说明、配套图注、文件名、数据协议、导出表和PDF元数据中。分享图片时应同时带上图注，不能把本图当成已完成实验的结果。

## 图中每一部分

| 元素 | 含义 |
| --- | --- |
| 一行 | 一种方法；七种方法的行位置在所有面板一致 |
| 四列 | SR成功率、CR碰撞率、BVR预算违规率、平均执行成本 |
| 圆形、三角、方形 | 普通障碍、窄通道、密集障碍三个场景 |
| 一个点 | 对应方法和场景的指标汇总；不是单次实验或某一时刻 |
| 横向短线及端帽 | 该点手动指定的假设估计范围，左端和右端可以不对称 |
| 方法颜色 | HD-MCRA蓝、RC-PPO橙、RA玫红；PPO两种设置用深浅紫色、CPPO两种设置用深浅绿色；浅色为beta L，深色为beta H |
| 上下轻微错开 | 为三个场景留出阅读空间，固定偏移，不表达数值或不确定性 |
| 上下箭头 | 成功率越高越好，碰撞和预算违规越低越好；成本须结合任务完成情况解释 |

图用于比较整体表现。它不能单独区分alpha和delta各自贡献，也不直接验证底层冻结或真实机器人部署。对应因果解释需要消融、跟踪结果和实机记录。

## 数据在哪里，如何推演

权威输入是 [data/assumed_intervals.jsonl](data/assumed_intervals.jsonl)，一行一个点：

```json
{"id":"P001","method":"HD-MCRA","scenario":"moderate","metric":"SR","estimate":85.0,"lower":77.4,"upper":90.8,"rationale":"该点的独立理由"}
```

`lower`、`upper`是端点，不是正负误差长度。例如P001为85.0%，左端77.4%，右端90.8%，绘图时对应左侧7.6、右侧5.8个百分点。脚本计算这两个长度属于坐标转换，不是生成原始端点。

率以百分数存储：85代表85%，不是0.85。成本沿用旧稿尚未核实物理意义的数值尺度，不声称它是焦耳、秒或米。

推演顺序为普通障碍P001–P028、窄通道P029–P056、密集障碍P057–P084。每批按方法分组，每组分别确定SR、CR、BVR和成本。每完成一个场景，就检查已有全部数据，再继续下一场景。三个审查时点的文件哈希记录在审查文档中。

每点的理由同时考虑已有均值、相邻方法、场景难度、其他指标与概率边界。没有随机采样，没有统一误差百分比，没有生成伪造训练seed或逐回合日志。端点数值是与这些条件相容的一种设计选择，不能从已有均值唯一推出，也不能保证未来真实结果落在其中。

在有限重复实验的尺度上选择了较宽且可重叠的范围，但并未把草稿中的样本数当成已验证事实；不标注SD、SE、95%CI或显著性。

## 本次一致性修正

旧工作簿和生成脚本中，RA的场景BVR为12%、20%、28%，等场景平均20%。最新论文主表原写22%，现统一为20%。

初次绘图时，84个中心值未变，只纠正了上述主表汇总值。随后在2026-10-23全文核查中，主结果图已换入`CCDC/latex/1020/main.tex`，工作簿的预算扫描和消融重复基准也已同步；本图的84个中心值和168个区间端点仍未改变。

当前草稿统一采用：每场景各方法共用参考预算，主表与消融表对应`z0/z_ref=1`；成本和MC按全部试验汇总到目标、物理碰撞或超时，预算超限只计入BVR、不提前结束评估。这是本次统一的假设口径，仍须真实日志验证，成本函数、绝对预算和单位尚未确定。完整记录见[数据一致性核查](../../CCDC/submission/1023数据一致性核查.md)。

跨场景主表是等场景平均，不等同于试验数量不等时的全试验合并比例。区间上下端不做简单平均来冒充总体区间。

## 文件分工

```text
data/
  assumed_intervals.jsonl       84个点及168个手动端点、逐点理由
  protocol.json                指标、单位、场景、颜色、数据性质、版式参数
scripts/
  plot_metrics_comparison.py   读数据、检查、绘图和导出；不生成端点
reviews/
  stage_reviews.md            各阶段发现、处理和验收
output/
  metrics_comparison_assumed.pdf  181 × 90 mm矢量图
  metrics_comparison_assumed.png  300 dpi预览
  assumed_point_estimates.csv     同一输入导出的方便查阅表
  推演明细.md                   完整逐点记录与等场景汇总
  caption.md                   中英图注
  data_audit.json              完整性、合法性和源值核对
  render_manifest.json         产物及输入哈希、版式信息
  pdf_check.png               从PDF渲染的复核图
```

导出CSV和Markdown是只读查阅副本；改值请改JSONL，再重绘，避免出现多份不一致的数据源。

## 重绘和检查

使用本机已有Conda base环境：`D:/codesoftware/Miniconda/python.exe`，Matplotlib 3.10.8。未安装新依赖。PDF复核使用现有Poppler。

从论文仓库根目录运行：

```powershell
& 'D:/codesoftware/Miniconda/python.exe' -B '绘图/20261023_多方法指标对比/scripts/plot_metrics_comparison.py' --check-only
& 'D:/codesoftware/Miniconda/python.exe' -B '绘图/20261023_多方法指标对比/scripts/plot_metrics_comparison.py'
```

默认重绘本目录output中的同名产物。保留当前版本时使用新的输出目录：

```powershell
& 'D:/codesoftware/Miniconda/python.exe' -B '绘图/20261023_多方法指标对比/scripts/plot_metrics_comparison.py' --output-dir '绘图/20261023_多方法指标对比/output_v2'
```

输出包括点估计、左右端点、理由、派生汇总和图注，区间长度始终读取数据，不写死在绘图函数中。排版控制在protocol.json的layout、methods、scenarios和metrics；字号、颜色、点形、范围可分别修改。

检查包括84点完整性、ID/组合唯一、所有数值有限、区间覆盖中心、率的0–100边界、SR+CR不超过100、区间不被坐标裁切、与旧中心值对应、文字不出画布、图例不与单位重叠。代数通过不代表物理或统计真实性通过。

## 后续换真实数据

1. 建立新数据目录，保留本次假设快照；将真实汇总和真实区间映射为相同84条记录。`rationale`可改为该点的试验范围、样本数、seed/地图组织和计算来源。
2. 在新protocol.json中将`data_status`设为`empirical`、`verify_legacy_centers`设为`false`。旧中心检查只用于本次继承假设值，真实结果无需匹配它们。
3. 填写`empirical_sources`、已核实的`sample_count_verified`、`interval_kind`、`interval_note`和对应置信水平（若采用CI）；这些是来源描述，不是仅改标签就把旧值变成实测。
4. 填入`empirical_caption_en`和`empirical_caption_zh`。明确每场景试验量、训练seed组织、统计单位、区间估计方法、预算比较方式、成本与失败计费。配置成本`unit`、`axis_label`，根据真实端点调整坐标`limits`和`ticks`。
5. 用`--data-dir`指向新目录、`--output-dir`指向新输出；同一绘图代码可复用点形、颜色、方法行和四面板布局，输出不带assumed后缀的图。程序只检查结构，不替代真实日志的统计计算和来源核验。
6. 分别更新论文主表、场景图与图注。若样本数量不均等，要明确等场景平均或按试验加权；若方法集合和场景集合改变，应相应修改固定84点布局与检查。

本图只需汇总与区间，计算真实区间所需的原始记录至少包括方法、场景、训练seed、地图/试验编号、成功/碰撞标记、实际累计成本、初始预算和终止原因。不必为这张图伪造或补画完整轨迹。

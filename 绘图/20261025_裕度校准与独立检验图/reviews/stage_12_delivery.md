# 全链路交付验收

交付对象是独立单栏绘图工程。计划38项按依赖推进，各项保留追加工作记录和验收依据；最后一项完成后由`final_audit.py --complete`封存38/38状态及交付清单。最终机器检查结果见`delivery_manifest.json`。

| 验收条件 | 证据 |
| --- | --- |
| 800校准+400留出、2400个误差逐项可追溯 | authored源文件、1200条point_reviews、cross_figure_audit源行逐项匹配 |
| 四个假设锚点和Q90算法一致 | statistics.json；stage_06、stage_09及计算器手算检查 |
| 先冻结校准再构造留出，不反向调阈值 | calibration_freeze.json记录held_out_exists=false，校准/协议哈希未变；stage_07/08 |
| 保留负值、等号、超限和指标取舍 | 23条负超支，留出47/54条单指标超限；各20组回顾及阶段报告 |
| 图间口径和假设边界明确 | cross_figure_contract.md、stage_09；不混用SR/CR/BVR及绝对Cost单位 |
| 完整单栏1×2及数值准确 | figure_numeric_audit逐阶梯计数、stage_11目视检查、export_inspection尺寸/字号 |
| PDF/SVG/600dpi PNG、图注、预览、清单完整 | output/final；88.9×52mm、最小6.5pt、全部ECDF和实际交点 |
| 独立复现通过 | reproduction_audit：PNG像素相同，PDF/SVG/PNG/图注/预览哈希相同，统计相同 |
| 更换数据有明确方法 | 项目README、data/README、real_data_replacement；真实值重新校准，不强配假设锚点 |
| 历史产物和只读来源不变 | source_manifest对127文件建立基线；delivery_manifest复核；绘图根README只追加 |
| 工作记录一致 | 计划每次更新后运行技能校验器；末项完成后再核对38条状态/证据 |

## 已闭环的问题

前期相对路径导入错误及过早标记008/009已有追加纠正，重新导入与全量匹配通过。全量阶段发现一对同条件汇总值重合，核对独立源行和不同假设命令ID后保留，不加噪伪装；强相关的人工特征已在说明中公开。试绘裁切、图例拥挤和阈值文字靠近曲线均通过仅改样式解决，历史版本保留。

没有真实采样、训练或远程操作；本轮没有改论文、插入图、编译或Git提交/推送。图只是展示假设校准和留出评估流程，不能当作真实独立实验验证或形式安全保证。

当前交付统计：α=0.18、δ=0.12m；留出成本覆盖353/400=88.25%，空间覆盖346/400=86.50%。全部来源、审查和重绘入口已齐备。

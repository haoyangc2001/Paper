# 当前真机对照数据（2026-10-29）

当前版本为v2，已撤销以旧86.7%为主要参照的乐观假设，重新检查窄通道与执行风险。完整/基线成功为35/50和32/50；Cost182/176，CR16%/20%，BVR14%/22%，MC精确值.086/.067m（表中.09/.07）。修订说明见[revision_v2.md](../../submission/hardware_comparison_20261029/revision_v2.md)。这是另一组有明确场景假设的合成示例，不是真机性能预测。
HD-MCRA和无校准RC-PPO各50次的内部合成示例。data_status=synthetic_example，empirical_validation=false；没有新增真实机器人测试。
论文入口：../../latex/1020/main.tex，5.6的tab:hardware_comparison和附录D。PDF：../../output/1020/main.pdf。

## 文件和修改顺序
- protocol.json：同条件比较、50次分母、继承配置、终止规则和指标定义。
- inference_ledger.jsonl：56条记录，原H01--H14/R01--R14保留为v1历史；H2-01--14/R2-01--14为当前28项输入，逐条revision_of关联，不覆盖旧条目。
- history/v1_hardware_comparison_results.json和v1_validation_report.json：首版结果与验证快照，不能作为当前主稿输入。
- hardware_comparison_results.json：当前唯一硬件统计结果入口；由已审查计数、均值及SD确定计算。
- validation_report.json：边界测试、计数、交叉关系、一二阶矩兼容性和保留文件哈希检查。
- ../../scripts/audit_hardware_comparison_20261029.py：只读派生与校验，输出到终端，不生成结果假设、不写文件。
- ../../submission/hardware_comparison_20261029/：来源基线、阶段审查、真实替换映射及最终交付审查。

## 统计语义
两方法各50，总100；不是每张照片各50。成功和终止失败互斥；BVR可与成功/失败重叠。Cost与MC包括所有终止结果，不只成功。
Cost为既有导航代理成本a.u.，不是焦耳。MC为每次最小物理净空的平均值，接触为0；内部.086/.067，表中舍入.09/.07。
比例区间为Wilson95 n50；连续均值区间为Student95 df49，由各自mean和sample SD计算。区间是合成输入的示例统计，不是实测证据，也不是单次数据范围。
构造性证人仅检查矩和尾部计数可兼容；不会输出或保存100条假传感器日志，不证明存在完整动力学轨迹。v2额外检查非超预算超时的成本至少216、成功试验的15m代理成本下界，并把低成本端分配至提前失败；不能用数学检查代替对任务难度的实测。

## 复核命令（在paper根目录）
```powershell
& 'D:/codesoftware/Miniconda/python.exe' -B 'CCDC/scripts/audit_hardware_comparison_20261029.py' --pilot
& 'D:/codesoftware/Miniconda/python.exe' -B 'CCDC/scripts/audit_hardware_comparison_20261029.py' --derive
& 'D:/codesoftware/Miniconda/python.exe' -B 'CCDC/scripts/audit_hardware_comparison_20261029.py' --check-tex
```
--derive只打印由台账计算的JSON。审查通过后再同步结果文件与论文，不能只改表里数值。
旧目录manuscript_completion_20261026仍是仿真/配置来源，但其hardware_results.json及protocol.hardware为30次历史版本。不要直接运行旧脚本hardware/results/full模式覆盖最新硬件结果。

真实替换时先核对日志/终止/预算交叉计数，再复算均值、SD、区间，最后同步表、正文、附录和结论。真实数据不要求保持当前优劣排序；样本相关时需要重新选择区间模型。

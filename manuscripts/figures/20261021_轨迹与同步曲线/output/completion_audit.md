# 最终验收对应表
最新数据v2，图形v3，正式入口output/final。下表逐项对应已批准计划的验收要求；全部数值均为假设。

| 任务 | 直接证据 |
| --- | --- |
| 001 | reference/source_manifest.json、README环境版本、目录与参考副本；复现报告核对5项原件哈希 |
| 002 | data/assumptions.md及冻结版本：参数、主表/消融/接口对照边界 |
| 003 | reference/visual_spec.md、完整第6页查看、output/reference_comparison.png |
| 004 | data/versions/v2/scenario.json与protocol.md：共同预算/几何/终止/代理成本 |
| 005 | schema.md、metadata.json：原始区间与N+1派生状态、事件trial和单位 |
| REVIEW-001 | output/reviews/01_design_review.md、ISSUE-001 |
| 006 | scripts/test_metrics.py、output/reproduction_check/test_results.txt：7项已知答案/错误注入检查 |
| 007 | data/pilot、pilot_attempt1、output/pilot_validation.json和失败报告 |
| REVIEW-002 | output/reviews/02_pilot_review.md、ISSUE-002 |
| 008 | v2/raw_rcppo.csv：66显式区间；full_validation_v2.json通过；失败/截断版本保留 |
| 009 | v2/raw_hd_mcra.csv：74显式区间；full_validation_v2.json通过 |
| 010 | data_audit_v1/v2.md、case_audit_v2.json、v1/v2快照及manifest；独立扫掠/事件/位移检查 |
| 011 | output/final/panels/trajectory.pdf和PNG；实体/足迹/delta、两条名义与实际路线、等比例米制 |
| 012 | output/final/panels/clearance、cost、tracking各PDF/PNG；同数据版本、各自事件和终止 |
| 013 | output/final/trajectory_sync_assumed.pdf和PNG，caption.md；宽184.2mm、矢量图文 |
| 014 | output/reviews/04_figure_review.md、visual_audit.md、pdf_qa_v3.json、print_size_preview.png；6问题closed |
| 015 | README复现与替换说明、reproduction_check/reproducibility_report.json、final_manifest.json及计划validator |

## 全局约束核对
- 原始点直接写入静态文件，脚本只计算派生量、验证和绘图；无策略训练/推理或仿真采集。
- 共同场景、初态、目标、足迹、接口和预算；原始实际速度独立设计，保留多时刻滞后/恢复。
- 不伪造统计覆盖或总体性能；假设标记、成本代理、较小超支与单例局限已在图注/README保留。
- 机身间隙与delta分开，线速度和角速度误差不混合，终止后无运动尾段。
- 三排曲线共享elapsed time，每方法的事件以自己的trial和时间对应地图。
- 复现结果直接比较数值文件、PNG像素和PDF字节；冻结输入及原稿/参考原件哈希不变。
- 旧图、旧数据、失败样本、审查和追加工作记录均保留；未修改论文或模板、未远程提交。


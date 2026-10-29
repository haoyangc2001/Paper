# 执行基线（TASK-001）
日期：2026-10-29。HEAD：5d0d35e98cce15652c3f07d19ba9d9fbf8806550。
开始时已跟踪文件干净，仅本次计划JSON及中文说明未跟踪；不提交、不清理。
当前PDF为10页；先前README的9页是历史状态，不作为本轮基线。已知会议全文上限8页，已有超页问题不在本次范围。
当前入口：CCDC/latex/1020/main.tex → CCDC/output/1020/main.pdf。
拟修改范围：5.6、附录D-A/D-C、活跃README、新数据及审核脚本、本计划、编译PDF。
不修改既有图片、仿真结果、方法正文、历史数据和台账。

Python：D:/codesoftware/Miniconda/python.exe（fitz可用，scipy未安装；不安装依赖，统计计算使用标准库及经验证常数）。
TeX：D:/app/latex/texlive/2026/bin/windows/pdflatex.exe，已存在。
原编译日志：无Overfull/undefined/duplicate；7处Underfull hbox，位于99、229、324（三处）、617、649附近。
原稿7张单栏图、2张单栏表，参考文献后另页附录。

旧脚本CCDC/scripts/audit_manuscript_completion_20261026.py：
- hardware()第244行起按旧26/30、184/44、0.11/0.07计算并写hardware_results.json，写死n=30。
- results汇总部分第272行写hardware_n=30。
- full_audit()第355行起要求旧184.0及167.6--200.4、0.084--0.136字符串。
禁止将这些生成/全量模式用于新稿。本次另写只读校验脚本。

旧结果：26成功、2定位中止、1超时、1接触；BVR=3（2成功+1超时），全部为历史合成示例。
旧protocol保留方法和配置契约；新protocol按明确字段继承并覆盖hardware部分。

## 保留文件校验值
- CCDC/data/manuscript_completion_20261026/hardware_results.json: 6461b9126d6b163fe0462f2c5850a5a2efff0e430fa229ce3987c360cf3c99f6
- CCDC/data/manuscript_completion_20261026/protocol.json: 691fe8270ea15fad78c3494158e45487729f1bd2f806a5bbc3e1a71ba0667cb1
- CCDC/latex/1020/IEEEtran.cls: c972aca108fda004c3514d63658e02816da2e54d9a1451e870b9bd970e003f55
- CCDC/latex/1020/references.bib: 4b4b458770c921cdfad84c5467256f409c3908d6c183ef9ec0bd2901cfa25870
- CCDC/images/1020/ablation_horizontal.pdf: 4351823d519805c900d48307991ee233890528a08e812d975e1ca87e2a6d03ea
- CCDC/images/1020/budget_sweep_horizontal.pdf: d1b1f4c432a3b092612cd87d75fa83fb2da4edef9b45b269032b240d49b2db72
- CCDC/images/1020/deployment_calibration_overview.pdf: d52b91217d81a6035b0bf95e0ce924c331e501525ec0b4c2ce92110ed46fdc3c
- CCDC/images/1020/figure_budget_sweep_v3.pdf: fd3df98cf283e2a5714ba2a84f58a1f421b983b2c0adb911cb1e1cdd8a231841
- CCDC/images/1020/figure_budget_sweep_v3.png: 7da39b4a6c7c4da211ab63b63d20d7c2233df293ddd18e3ccfdf2e85a713c2fe
- CCDC/images/1020/figure_metrics_grid_single_column_v3.pdf: 601b69c8e3e34a55dab4d043815d7bacec81ad25a015f723d208470c9ee5ccd1
- CCDC/images/1020/framework_overview_final.png: 7a5414042068ab0391edd0ba1e9ae5acc091fc72a25644c718453ba674d3ae83
- CCDC/images/1020/margin_calibration.pdf: d959a7d76a49830b8fbdae13c1374fa4bfd8ca397556da9c9917847e6f288cf4
- CCDC/images/1020/metrics_comparison_assumed.pdf: 9c93488178a96d6a44da8902da6eb052a4bf4cdc531f23f8d776c4b4474cee24
- CCDC/images/1020/metrics_comparison.pdf: 78b6d59d9973ed01125253c07d5e73f67964baf6eb9d77b7b4f2a176e4b0f7af
- CCDC/images/1020/office_sequence_candidate_A_numbered.jpg: 934f1ea9d00d5c218e1e99710a9b0d3c95bef138f7813d6deebac3a49ab62326
- CCDC/images/1020/office_shortest_path_sequence_final.png: 7386317de92b2c086f73bc8e48f47c74398458dfb648fbb21c48a4d6852fd1e9
- CCDC/images/1020/trajectory_sync.pdf: 5e51a1741ade6af2211ed8717e9e8279143990aa3d09a12d680565470f411dad

# 阶段复查D：表格输入冻结
专用只读脚本的小批和全量审查通过；validation_report.json为验证证据。
28条独立输入记录覆盖两方法的成功/失败、BVR各交叉格、Cost/MC均值与样本SD。
10个中心、10对95%区间由这些输入确定计算，无手填端点；Wilson n50，Student t df49、临界值2.0095752371292397。
小批修正：25/50的测试预期常数录入错误，改为p=.5独立闭式公式；统计函数和作者输入未改变。修正前失败和修正后通过均已保留于计划记录。

表格显示按SR、Cost、CR、BVR、MC：
- HD-MCRA：86.0、186.0、4.0、10.0、0.11。
- RC-PPO：80.0、181.0、8.0、18.0、0.08。
MC内部保留.108/.082，差值.026不可从两位显示值反推。
样本规模50/方法；不和历史30例、三场景900例合并。
来源性质：data_status=synthetic_example；empirical_validation=false。照片不证明这些计数，区间不表示已实测不确定性。
不能由此声称显著差异、单裕度因果贡献或普遍安全保证。

冻结哈希：

Path : C:\Users\smile\Desktop\paper\CCDC\data\hardware_comparison_20261029\hardware_comparison_results.json
Hash : 4307758FDB996AFC19DB3109874504AA47472842E351DE2EE64B8B6A8AE0253C

Path : C:\Users\smile\Desktop\paper\CCDC\data\hardware_comparison_20261029\inference_ledger.jsonl
Hash : 84667B1CFAC36D97064E47BD8E03C8BF08A4993098851E5236E5CCC7A34B8A1C

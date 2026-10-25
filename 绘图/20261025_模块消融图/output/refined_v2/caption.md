# Figure caption

## English

Component ablation at the common reference budget (all values are assumed data). Blue filled circles show full HD-MCRA; colored hollow markers show the indicated deletion variant (both margins disabled is RC-PPO), with the same frozen tracker. The alpha/delta columns indicate enabled (check) or disabled (cross) margins; full HD-MCRA enables both, and the both-disabled row is RC-PPO. Header values are the full-method references; labels at hollow points are deletion values. Segments connect aggregate centers, not uncertainty intervals; each row reuses the same full-method reference. The changes relative to the full method are +13.4 percentage points in BVR without alpha and +11.3 percentage points in CR without delta. Cost averages all trials up to termination and must be interpreted with SR. Axes use local ranges.

## 中文

相同参考预算下的模块消融（全部为假设数据）。蓝色实心圆为完整HD-MCRA，彩色空心点为对应删模块版本；两者均关闭对应RC-PPO，均使用同一冻结跟踪器。左侧α/δ列中勾表示启用、叉表示关闭；完整HD-MCRA两者均启用，双叉行对应RC-PPO。栏顶蓝色数值为完整方法参照，空心点旁为删除配置数值。连接线表示汇总中心值差异，不是误差区间；每一行重复使用同一完整方法参照。当前假设值中，去α的BVR变化为+13.4个百分点，去δ的CR变化为+11.3个百分点。成本统计所有试验截至终止的累计执行成本，需要结合成功率解释。坐标采用局部范围。

## Reading notes

- Panel order: SR, Cost, BVR, CR. Rates are percentages; highlighted differences use percentage points.
- Three deletion rows are categories, not time or a parameter sweep.
- Full and RC-PPO use equal-scene means at z0/zref = 1. Cost units remain draft units.
- 12 segments and 24 markers reuse 16 distinct centers; no measured confidence intervals are supplied.
- Single-margin settings need not improve monotonically over the no-margin setting.
- MC and the joint-space interface comparison remain in the existing table.
- Replace assumptions only with verified trial/seed/map data and matching statistical definitions.

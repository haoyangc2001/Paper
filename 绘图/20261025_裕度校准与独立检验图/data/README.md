# 最终数据与字段

最终绘图只读`paired_rollouts.csv`。共1200行：CAL-0001至CAL-0800为校准，TEST-0001至TEST-0400为留出。全部`data_status=hypothetical`，是明确构造的示例汇总值，不能描述为实测配对日志。

## 字段定义

| 字段 | 类型/单位 | 含义与约束 |
| --- | --- | --- |
| pair_id | 唯一字符串 | 一组配对运行的假设记录ID，一行只计一个样本 |
| split | calibration / held_out | 预先指定的数据组；留出不参与选阈值 |
| condition_id | 字符串 | 场景+扰动组合，例如S_B |
| scenario_id | S/A/L/T | 直行变速/宽弧/交替转弯/停走转向 |
| perturbation_profile_id | B/F/D/N/M | 基准/低摩擦/延迟/噪声/混合扰动 |
| duration_s | s | 同一对名义和执行的公共观察窗口，S12/A16/L20/T14 |
| command_id | 唯一字符串 | 假设的配对命令标识；没有对应真实时序命令文件 |
| controller_id | 字符串 | `assumed_frozen_velocity_tracker_v1`，不是实际权重哈希 |
| J_nom | 示例成本单位 | 名义成本，严格大于0 |
| J_exec | 同上 | 执行成本，本版由J_nom与手写超支按定义计算，严格大于0 |
| cost_overrun | 无量纲比值 | (J_exec−J_nom)/J_nom，可为负；0.18在图中画18% |
| max_deviation_m | m | 时间对齐后名义/执行位置的最大平面差，非负；0.12就是12cm |
| data_status | hypothetical | 保留数据性质，不靠改字符串把假设变成实测 |

扰动参数在`protocol.json`，所有数值均为假设设置。20个条件在校准各40行、留出各20行。行顺序按条件循环，是编排顺序，不是机器人连续时序。不能从这些汇总量复原真实轨迹或噪声序列。

## 文件分工

- `authored/*.txt`：人工逐行输入`J_nom|超支百分比|最大偏差毫米|成本判断|几何判断`；`#`行是每20组回顾。注意这与最终CSV单位不同。
- `calibration_draft.csv`、`held_out_draft.csv`：分批导入的审查工作表；不是绘图脚本入口。
- `calibration_frozen.csv`、`calibration_freeze.json`：800组校准及冻结阈值/口径/哈希；冻结发生在第一条留出写入前。
- `paired_rollouts.csv`：已验收合并版，唯一绘图输入；SHA256为`292910afc8e2ce728be73e6fb58c14f4924fa3b3ce2eba9121bdf66d3022b03b`。
- `statistics.json`：从最终CSV重算的两组×两指标完整ECDF、范围、中位数、Q90、阈值、等号/超限/负值计数、边际及内部联合覆盖。
- `calibration_statistics.json`、`held_out_statistics.json`：冻结/阶段审查时的分组统计快照。
- `split_manifest.csv`、`condition_catalog.csv`：预分配ID及条件，不是采集证据。
- `protocol.json`、`record_schema.json`、`reference_anchors.json`、`metadata.json`：协议、字段契约、旧稿假设锚点、来源局限。

## 算法与计数例子

经验Q90为排序后第ceil(0.9n)项，800组取第720项；不用插值，也不用共形预测的n+1修正。中位数单独使用普通中位数算法，不能误把第400项当偶数样本中位数。

成本阈值0.18，校准727条≤阈值，其中8条恰等于阈值；空间阈值0.12m，校准722条≤阈值，其中8条等号。所以校准ECDF阈值处不恰好90%。

留出成本353/400=88.25%，空间346/400=86.50%；每多覆盖一条只增加0.25个百分点。联合覆盖336条，只有成本覆盖17条，只有空间覆盖10条，两项均超37条，总和400；这不等于导航成功/碰撞事件。

## 修改数据后的正确路径

1. 保留当前CSV、冻结清单和逐组判断。新建版本文件，不直接覆盖已验收数据。
2. 逐ID追加旧值、新值及理由，并检查同条件、前后多个案例、另一指标和其他图。不要为了图线光滑或覆盖接近90%调值。
3. 修改J_nom、J_exec、cost_overrun时确保三者定义一致。只改几何值时检查m/mm，不把毫米直接写入m列。若改变校准值，对新版本重新选阈值并冻结，不能混用旧哈希。
4. 用`compute_statistics.py --data 新CSV --output 新统计JSON`重算；该工具只检查基础字段与定义式，不能替代人工来源审查或完整冻结校验。
5. 完整验证工具`validate_data.py --final`还绑定本版manifest、冻结文件和逐组审查。新版本应显式切换或扩展这些输入并保留旧版；不要删掉校验来“通过”。
6. 用`plot_margin_calibration.py --data 新CSV --statistics 新统计JSON --output 新目录`重绘。曲线、阈值、覆盖率和图注中的数字自动更新，不能手改图里的百分比。

真实数据需要额外来源字段和日志核验；当前基础校验、统计元数据及图注均显式针对假设数据，真实替换需要同步升级这些入口。详见[替换说明](../reviews/real_data_replacement.md)。

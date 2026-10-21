# 数据契约 v1（Assumed data）
所有文件UTF-8；小数点为点；禁止NaN/空值。raw列名含单位。两方法分别为raw_rcppo.csv、raw_hd_mcra.csv；试验ID为two_bends_01_rcppo/hd_mcra，由文件名和scenario_id确定，metadata.json固定声明data_status=assumed。

## 原始区间记录
t_s,v_cmd_x_mps,v_cmd_y_mps,yaw_rate_cmd_radps,v_exec_x_mps,v_exec_y_mps,yaw_rate_exec_radps
七列均为显式逐点编写；body速度；每行是[t,t+0.2)区间的代表值。t从0开始严格递增0.2。yaw初值0。速度是权威记录，不另编独立位置；不在脚本中随机造数。t=0代表第一个区间平均动作，不要求瞬时静止。

## 派生状态记录
derived_rcppo.csv与derived_hd_mcra.csv：
t_s；x_nom_m,y_nom_m,yaw_nom_rad,x_exec_m,y_exec_m,yaw_exec_rad；
clearance_nom_m,clearance_exec_m,path_deviation_m；
tracking_error_mps,yaw_error_radps；
cost_nom_step,cost_exec_step,cost_charged_step；
J_nom,J_exec,J_charged,z_internal,z_actual；
budget_violation,terminal。
N+1状态行；前N行误差对应从该时刻开始的区间，成本增量也是该区间；累计量是到该时刻为止。最后行无动作，误差/增量空值（只允许派生终止行），累计量保留。绘图误差只用前N行。终止在区间末判定。
所有累计量double精度计算，CSV保存至少9位有效小数；比较绝对容差1e-8用于计算回算，时间容差1e-8 s。原始数据最多3位小数，积分误差不假装传感器精度。

## 事件与来源
events.csv：event_id,method,trial_id,t_s,sample_index,x_m,y_m,description。A/B来自首次越过相应x平面，源自执行状态采样，不插值伪造新观测；没有发生的事件不虚构。
metadata.json记录两方法trial、原始文件、假设属性及场景版本。revision_notes.md记录区段设计、问题与修改。data/versions/v1保存完整可复现快照。
图例和图注声明assumed；名义轨迹是同命令理想跟踪，不是另一份独立测量。

